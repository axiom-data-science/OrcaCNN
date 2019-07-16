#!/usr/bin/env python3
# coding=utf-8

import argparse
import logging
import logging.config
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_and_end_time_orca(test_path):
    '''Determine the start and end call of orcas in the acoustic sample'''
    i = 1
    dur = 0
    flag = 0
    path = sorted(
        os.listdir(test_path),
        key=lambda x: int(
            os.path.splitext(x)[0]))
    N = sum(len(files) for _, _, files in os.walk(test_path))
    for elem, next_elem in zip(path, path[1:] + [path[0]]):
        f_num = int(os.path.splitext(elem)[0])
        f_num_next = int(os.path.splitext(next_elem)[0])
        init_call = f_num + 1
        if flag == 0:
            print(f"Start time of orca call {i}: {init_call}")
            flag = 1
        if f_num + 1 == f_num_next:
            dur = dur + 1
            continue
        if f_num + 1 != f_num_next:
            print(f"End time of orca call {i}: {f_num+1}")
        flag = 0
        i += 1
    logger.info(f"Total {i-1} orca calls found with duration: {N} seconds")


def main(args):
    test_path = args.classpath
    start_and_end_time_orca(test_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Function to match template in detected orca spectrograms")
    parser.add_argument(
        '-c',
        '--classpath',
        type=str,
        help='directory with list of orca calls',
        required=True)

    args = parser.parse_args()

    main(args)
