import json
import subprocess
import sys
import unittest

from tumeur.ai import recommend_action
from tumeur.game import TumorGame, make_config


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
        self.assertNotIn("n_cells", action)

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
