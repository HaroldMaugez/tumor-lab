import json
import pickle
import subprocess
import sys
import unittest

from tumeur.ai import load_policy_net, recommend_action
from tumeur.game import TumorGame, make_config
from tumeur.neural_net import check_gradients


class GameContractTests(unittest.TestCase):
    def test_strict_mode_preserves_avascular_regime(self):
        game = TumorGame(make_config("in_vitro_strict", seed=11))
        obs = game.observe()
        self.assertEqual(obs["model_regime"], "avasculaire_validated")
        self.assertEqual(obs["vascular_activity"], 0.0)

    def test_vascular_action_changes_exposed_state(self):
        game = TumorGame(make_config("vascular_human", seed=11))
        before = game.vascular.activity
        result = game.step({"type": "antiangiogenic", "intensity": 1.0})
        self.assertTrue(result.observation["vascular_activity"] <= before)
        self.assertEqual(result.observation["model_regime"], "vascular_scenario")

    def test_ai_is_observation_limited_and_explainable(self):
        game = TumorGame(make_config("vascular_human", seed=3))
        action = recommend_action(game)
        self.assertIn(action["type"], {"wait", "irradiate", "biopsy", "po2_imaging", "antiangiogenic"})
        self.assertTrue(action["reason"])
        self.assertTrue(action["reason_en"])
        self.assertNotIn("n_cells", action)
        self.assertTrue(action["trained"])
        probs = action["probabilities"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)

    def test_score_is_emitted_only_after_terminal_result(self):
        game = TumorGame(make_config("tutoriel", seed=2, max_days=1))
        self.assertEqual(game.observe()["score"], 0.0)
        result = game.step({"type": "wait"})
        self.assertTrue(result.done)
        self.assertIsInstance(result.observation["score"], float)

    def test_irradiation_turn_completes(self):
        game = TumorGame(make_config("standard", seed=2))
        result = game.step({"type": "irradiate", "dose_gy": 2.0})
        self.assertIn("radiation", result.info)

    def test_all_ui_organs_can_start_and_irradiate(self):
        for organ in ("generique_tardif", "poumon", "sein", "colorectal", "cerveau"):
            with self.subTest(organ=organ):
                game = TumorGame(make_config("standard", seed=2, organ=organ))
                result = game.step({"type": "irradiate", "dose_gy": 2.0})
                self.assertIn("radiation", result.info)

    def test_game_state_can_be_persisted(self):
        game = TumorGame(make_config("standard", seed=2))
        game.step({"type": "wait"})
        restored = pickle.loads(pickle.dumps(game))
        self.assertEqual(restored.turn, game.turn)
        self.assertEqual(restored.observe()["day"], game.observe()["day"])


class NeuralNetContractTests(unittest.TestCase):
    def test_analytical_backprop_matches_finite_differences(self):
        res = check_gradients(seed=42)
        self.assertTrue(res["passed"], msg=str(res))
        self.assertLess(res["supervised_max_rel_error"], 1e-5)
        self.assertLess(res["a2c_max_rel_error"], 1e-5)

    def test_trained_weights_load_and_respect_action_masks(self):
        net, meta = load_policy_net()
        self.assertTrue(meta.get("trained"))
        self.assertEqual(net.num_parameters(), 5575)

        game_avasc = TumorGame(make_config("standard", seed=5))
        act = recommend_action(game_avasc)
        self.assertEqual(act["probabilities"]["antiangiogenic"], 0.0)
        self.assertGreaterEqual(len(act["top_signals_en"]), 2)


class WebAppOnboardingTests(unittest.TestCase):
    def setUp(self):
        from wsgi import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_one_click_guest_tutorial_and_language_switch(self):
        r_home = self.client.get("/")
        self.assertEqual(r_home.status_code, 200)
        self.assertIn(b"New to biology or oncology?", r_home.data)

        r_fr = self.client.get("/set-lang/fr", follow_redirects=True)
        self.assertEqual(r_fr.status_code, 200)
        self.assertIn("Tutoriel guidé".encode("utf-8"), r_fr.data)

        # Guest can open tutorial directly with zero sign-up wall
        r_play = self.client.get("/play/tutoriel")
        self.assertEqual(r_play.status_code, 200)

        # Reset, consult Neural Net advisor, and step without leaking truth
        r_reset = self.client.post("/api/game/reset", json={"scenario": "tutoriel", "seed": 7})
        self.assertEqual(r_reset.status_code, 200)
        self.assertTrue(r_reset.get_json()["ok"])

        r_ai = self.client.post("/api/game/ai")
        self.assertEqual(r_ai.status_code, 200)
        ai_json = r_ai.get_json()
        self.assertTrue(ai_json["ok"])
        self.assertIn("probabilities", ai_json["action"])

        r_step = self.client.post("/api/game/step", json={"action": {"type": "po2_imaging"}})
        self.assertEqual(r_step.status_code, 200)
        step_json = r_step.get_json()
        self.assertNotIn("truth", step_json)
        self.assertIsNotNone(step_json["observation"]["last_po2_imaging"])

    def test_instant_cached_validation_endpoint(self):
        r_tests = self.client.get("/api/tests/run")
        self.assertEqual(r_tests.status_code, 200)
        payload = r_tests.get_json()
        self.assertEqual(payload["total"], 32)
        self.assertEqual(payload["passed"], 31)

    def test_cookieless_iframe_player_token_advances_turns(self):
        from wsgi import app

        # Simulate a cross-origin iframe where third-party cookies are completely blocked:
        # every HTTP call uses a fresh client with an empty cookie jar.
        c1 = app.test_client()
        r_reset = c1.post("/api/game/reset", json={"scenario": "tutoriel", "seed": 9})
        self.assertEqual(r_reset.status_code, 200)
        reset_json = r_reset.get_json()
        self.assertTrue(reset_json["ok"])
        token = reset_json["player_token"]
        self.assertTrue(token)
        self.assertEqual(reset_json["observation"]["turn"], 0)

        c2 = app.test_client()
        r_step1 = c2.post(
            "/api/game/step",
            headers={"X-Player-Token": token, "X-Lang": "fr"},
            json={"action": {"type": "po2_imaging"}},
        )
        self.assertEqual(r_step1.status_code, 200)
        s1 = r_step1.get_json()
        self.assertTrue(s1["ok"])
        self.assertEqual(s1["observation"]["turn"], 1)

        c3 = app.test_client()
        r_step2 = c3.post(
            "/api/game/step",
            json={"player_token": token, "action": {"type": "irradiate", "dose_gy": 2.0}},
        )
        self.assertEqual(r_step2.status_code, 200)
        s2 = r_step2.get_json()
        self.assertTrue(s2["ok"])
        self.assertEqual(s2["observation"]["turn"], 2)

        c4 = app.test_client()
        r_ai = c4.post("/api/game/ai", headers={"X-Player-Token": token}, json={})
        self.assertEqual(r_ai.status_code, 200)
        self.assertTrue(r_ai.get_json()["ok"])

        c5 = app.test_client()
        r_exp = c5.get(f"/api/game/export?token={token}")
        self.assertEqual(r_exp.status_code, 200)
        self.assertEqual(r_exp.get_json()["observation"]["turn"], 2)


class ServerContractTests(unittest.TestCase):
    def test_provenance_is_explicitly_scenario_level(self):
        process = subprocess.Popen(
            [sys.executable, "-u", "server.py"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            process.stdin.write(json.dumps({"cmd": "reset", "scenario": "vascular_human"}) + "\n")
            process.stdin.write(json.dumps({"cmd": "provenance"}) + "\n")
            process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            process.stdin.flush()
            json.loads(process.stdout.readline())
            provenance = json.loads(process.stdout.readline())
            self.assertEqual(provenance["vascular"]["exchange_o2_per_s"]["status"], "SCENARIO")
        finally:
            process.wait(timeout=10)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()

    def test_ping_advertises_v2_features(self):
        process = subprocess.Popen(
            [sys.executable, "-u", "server.py"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            process.stdin.write(json.dumps({"cmd": "ping"}) + "\n")
            process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            process.stdin.flush()
            ping = json.loads(process.stdout.readline())
            self.assertEqual(ping["version"], "2.0")
            self.assertIn("vascular_human", ping["features"])
        finally:
            process.wait(timeout=10)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()
