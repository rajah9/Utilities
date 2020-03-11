from win32api import SetCursorPos, mouse_event, GetCursorPos
from win32con import MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
from Util import Util

class InputUtil(Util):
    def __init__(self):
        super(InputUtil, self).__init__(null_logger=True)


class MouseUtil(InputUtil):
    def move_to(self, x:int, y:int):
        pt = (x, y)
        self.logger.debug(f'moving to {x}, {y}')
        SetCursorPos(pt)

    def click_on(self, x:int, y:int) -> None:
        """
        This clicks on a position without moving the mouse.
        :param x:
        :param y:
        :return:
        """
        mouse_event(MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0)
        self.logger.debug('click!')

    def click(self, x:int, y:int) -> None:
        """
        click moves the mouse and clicks in that position.
        :param x:
        :param y:
        :return:
        """
        self.move_to(x, y)
        self.click_on(x, y)

    def get_cursor_pos(self):
        pt = GetCursorPos()
        self.logger.debug(f'cursor position is {pt}')
        return pt


