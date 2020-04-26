import unittest

from nl2prog.gin import workspace


class TestWorkspace(unittest.TestCase):
    def test_use_workspace(self):
        with workspace.use_workspace():
            workspace.put("key", "value")
            self.assertEqual("value", workspace.get("key"))
            self.assertEqual(None, workspace.get("not-found"))
        self.assertEqual(None, workspace.get("key"))


if __name__ == "__main__":
    unittest.main()
