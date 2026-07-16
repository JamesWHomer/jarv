import unittest

from jarv.jsonutil import iter_json_objects, salvage_json_object


class SalvageJsonObjectTests(unittest.TestCase):
    def test_fenced_object_is_recovered(self):
        text = 'Here are the arguments:\n```json\n{"command": "echo hi"}\n```\n'
        self.assertEqual(salvage_json_object(text), {"command": "echo hi"})

    def test_object_with_trailing_prose_is_recovered(self):
        text = '{"path": "a.txt"} — reading the file now.'
        self.assertEqual(salvage_json_object(text), {"path": "a.txt"})

    def test_object_after_leading_prose_is_recovered(self):
        text = 'Sure! {"query": "jarv"}'
        self.assertEqual(salvage_json_object(text), {"query": "jarv"})

    def test_two_top_level_objects_are_ambiguous(self):
        text = '{"a": 1} {"b": 2}'
        self.assertIsNone(salvage_json_object(text))

    def test_array_is_not_salvaged(self):
        # The first decodable value must be an object; nested lookups would
        # be guessing which dict the model meant.
        self.assertIsNone(salvage_json_object('[{"a": 1}]'))

    def test_garbage_and_truncated_json_stay_unparseable(self):
        self.assertIsNone(salvage_json_object("not json at all"))
        self.assertIsNone(salvage_json_object('{"longform": "partial'))
        self.assertIsNone(salvage_json_object(""))

    def test_braces_inside_json_strings_survive(self):
        text = 'note: {"msg": "use {braces} here"} done'
        self.assertEqual(salvage_json_object(text), {"msg": "use {braces} here"})

    def test_stray_braces_in_prose_do_not_block_recovery(self):
        text = 'weird {brace then the real thing {"ok": true}'
        self.assertEqual(salvage_json_object(text), {"ok": True})


class IterJsonObjectsTests(unittest.TestCase):
    def test_yields_nested_objects_for_shape_scans(self):
        # The auditor scans for a verdict-shaped dict even inside an outer
        # envelope; every brace must be tried.
        values = list(iter_json_objects('{"outer": {"allow": true}}'))
        self.assertIn({"outer": {"allow": True}}, values)
        self.assertIn({"allow": True}, values)

    def test_skips_undecodable_braces(self):
        values = list(iter_json_objects('{nope} {"a": 1}'))
        self.assertEqual(values, [{"a": 1}])


if __name__ == "__main__":
    unittest.main()
