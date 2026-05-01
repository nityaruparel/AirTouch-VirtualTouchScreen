import pyautogui

def get_screen_size():
    width, height = pyautogui.size()
    return width, height