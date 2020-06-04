import signal
from utils.logger import log
from utils.action_enum import ActionEnum


class SignalHandler(object):
    def __init__(self, sigint_action, sighup_action):
        self._sigint_action = sigint_action
        self._sighup_action = sighup_action
        self._got_sighup = False
        self._got_sigint = False
        self._already_hooked_up = False
        self._hookup_handler()

    def signal2action(self):
        if self._is_sighup_up():
            return self._sighup_action
        if self._is_sigint_up():
            return self._sigint_action
        return 0  # refers to _ActionEnum.NONE

    def _signal_handler(self, signum, frame):
        if signum == signal.SIGINT:
            self._got_sigint = True
        elif signum == signal.SIGHUP:
            self._got_sighup = True
        else:
            pass

    def _hookup_handler(self):
        if self._already_hooked_up:
            log.fatal('Tried to hookup signal handlers more than once.')
        self._already_hooked_up = True
        signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _un_hook_handler(self):
        if self._already_hooked_up:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGHUP, signal.SIG_DFL)
            self._already_hooked_up = False

    def _is_sigint_up(self):
        result = self._got_sigint
        self._got_sigint = False
        return result

    def _is_sighup_up(self):
        result = self._got_sighup
        self._got_sighup = False
        return result


signal_handler = SignalHandler(ActionEnum.STOP, ActionEnum.CHECKPOINT)

if __name__ == '__main__':
    from time import sleep
    action = SignalHandler(ActionEnum.STOP, ActionEnum.CHECKPOINT)
    ctr = 0
    while True:
        if action.signal2action() == ActionEnum.STOP:
            log.info('exit')
            break
        if ctr % 1000 == 0:
            print('still running')
            sleep(0.1)
        ctr += 1