import unittest

from jarv.shell import CommandResult, truncate_model_output


class ShellOutputLimitTests(unittest.TestCase):
    def test_command_output_is_middle_truncated_for_model(self):
        output = truncate_model_output("a" * 50 + "MIDDLE" + "z" * 50, 100)

        self.assertIn("tool output truncated to 100 characters", output)
        self.assertIn("characters omitted from the middle", output)
        self.assertTrue(output.startswith("a"))
        self.assertTrue(output.endswith("z"))
        self.assertNotIn("MIDDLE", output)

    def test_command_result_keeps_status_inside_limit(self):
        result = CommandResult("cmd", "x" * 200, "", 2)

        output = result.to_model_output(120)

        self.assertIn("command output truncated", output)
        self.assertIn("[exit code 2]", output)


if __name__ == "__main__":
    unittest.main()
