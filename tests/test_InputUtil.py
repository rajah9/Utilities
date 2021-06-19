from unittest import TestCase

from InputUtil import MouseUtil


class TestMouseUtil(TestCase):
    def setUp(self) -> None:
        self.mu = MouseUtil()

    def test_move_to(self):
        x, y = 100, 200
        expected = (x, y)
        self.mu.move_to(x, y)
        actual = self.mu.get_cursor_pos()
        self.assertEqual(expected, actual)

    def test_click_on(self):
        orig_point = self.mu.get_cursor_pos()
        x, y = 300, 400
        self.mu.click_on(x, y)
        actual = self.mu.get_cursor_pos()
        self.assertEqual(orig_point, actual)

    def test_click(self):
        x, y = 300, 400
        expected = (x, y)
        self.mu.click(x, y)
        actual = self.mu.get_cursor_pos()
        self.assertEqual(expected, actual)

    def test_get_cursor_pos(self):
        x, y = 300, 250
        expected = (x, y)
        self.mu.move_to(x, y)
        actual = self.mu.get_cursor_pos()
        self.assertEqual(expected, actual)


    