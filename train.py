import datetime
import signal
import subprocess
import sys

import tensorflow as tf
# import win32job

from dataset import DataSet
from model import Seq2Seq
from summary_writer import TrainSummaryWriter

import warnings

# import winerror
# import win32api
# import win32job
#
# g_hjob = None

#
# def create_job(job_name='', breakaway='silent'):
#     hjob = win32job.CreateJobObject(None, job_name)
#     if breakaway:
#         info = win32job.QueryInformationJobObject(hjob,
#                                                   win32job.JobObjectExtendedLimitInformation)
#         if breakaway == 'silent':
#             info['BasicLimitInformation']['LimitFlags'] |= (
#                 win32job.JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)
#         else:
#             info['BasicLimitInformation']['LimitFlags'] |= (
#                 win32job.JOB_OBJECT_LIMIT_BREAKAWAY_OK)
#         win32job.SetInformationJobObject(hjob,
#                                          win32job.JobObjectExtendedLimitInformation, info)
#     return hjob

#
# def assign_job(hjob):
#     global g_hjob
#     hprocess = win32api.GetCurrentProcess()
#     try:
#         win32job.AssignProcessToJobObject(hjob, hprocess)
#         g_hjob = hjob
#     except win32job.error as e:
#         if (e.winerror != winerror.ERROR_ACCESS_DENIED or
#                 sys.getwindowsversion() >= (6, 2) or
#                 not win32job.IsProcessInJob(hprocess, None)):
#             raise
#         warnings.warn('The process is already in a job. Nested jobs are not '
#                       'supported prior to Windows 8.')


# def limit_memory(memory_limit):
#     if g_hjob is None:
#         return
#     info = win32job.QueryInformationJobObject(g_hjob,
#                 win32job.JobObjectExtendedLimitInformation)
#     info['ProcessMemoryLimit'] = memory_limit
#     info['BasicLimitInformation']['LimitFlags'] |= (
#         win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY)
#     win32job.SetInformationJobObject(g_hjob,
#         win32job.JobObjectExtendedLimitInformation, info)

if __name__ == '__main__':
    # limit_memory(40 * 1_000 * 1_000 * 1_000)
    sys.setrecursionlimit(15000)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dataset = DataSet(10, 80_000, 10_000, True)
    model = Seq2Seq(128, 320, dataset)
    summary_writer = TrainSummaryWriter("logs")
    model.set_summary_writer(summary_writer)

    logs_dir = "logs"
    current_date = datetime.date.today()
    current_date = current_date.strftime("%d-%m-%Y")
    logs_dir += "-" + current_date
    # tensorboard_process = subprocess.Popen(f"tensorboard --logdir={logs_dir}")
    # try:
    model.train(10, True)
    model.load_last("12-12-2020")
    print(model.evaluate_sentence("given an array of numbers what is first half of the given array"))
    # tensorboard_process.send_signal(signal.CTRL_C_EVENT)
    # print("Tensorboard killed")
    # except Exception as e:
    # tensorboard_process.send_signal(signal.CTRL_C_EVENT)
    # tensorboard_process.kill()
    print("Tensorboard killed")
    raise e
