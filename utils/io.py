# Copyright (c) OpenMMLab. All rights reserved.
# from mmcv == 1.6.2 but edited
# From mmcv utils io.py
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Union

from .file_handler import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler

import abc
import os

FileLikeObject = Union[TextIO, StringIO, BytesIO]

file_handlers = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
    'pickle': PickleHandler(),
    'pkl': PickleHandler()
}
def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def load(file: Union[str, Path, FileLikeObject],
         file_format: Optional[str] = None,
         file_client_args: Optional[Dict] = None,
         **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Note:
        In v1.3.16 and later, ``load`` supports loading data from serialized
        files those can be storaged in different backends.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Examples:
        >>> load('/path/of/your/file')  # file is storaged in disk
        >>> load('https://path/of/your/file')  # file is storaged in Internet
        >>> load('s3://path/of/your/file')  # file is storaged in petrel

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    handler = file_handlers[file_format]
    f: FileLikeObject
    if isinstance(file, str):
        if handler.str_like:
            with open(file, encoding='utf-8') as f:
                file_text_buf = f.read()
            with StringIO(file_text_buf) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with open(file, 'rb') as f:
                file_buf = f.read()
            with BytesIO(file_buf) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj: Any,
         file: Optional[Union[str, Path, FileLikeObject]] = None,
         file_format: Optional[str] = None,
         file_client_args: Optional[Dict] = None,
         **kwargs):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    Note:
        In v1.3.16 and later, ``dump`` supports dumping data as strings or to
        files which is saved to different backends.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk
        >>> dump('hello world', 's3://path/of/your/file')  # ceph or petrel

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None')
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')
    f: FileLikeObject
    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(file, str):
        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                mkdir_or_exist(os.path.dirname(file))
                with open(file, 'w', encoding='utf-8') as f_w:
                    f_w.write(f.getvalue())
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                mkdir_or_exist(os.path.dirname(file))
                with open(file, 'wb') as f_w:
                    f_w.write(f.getvalue())
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def _register_handler(handler: BaseFileHandler,
                      file_formats: Union[str, List[str]]) -> None:
    """Register a handler for some file extensions.

    Args:
        handler (:obj:`BaseFileHandler`): Handler to be registered.
        file_formats (str or list[str]): File formats to be handled by this
            handler.
    """
    if not isinstance(handler, BaseFileHandler):
        raise TypeError(
            f'handler must be a child of BaseFileHandler, not {type(handler)}')
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if not is_list_of(file_formats, str):
        raise TypeError('file_formats must be a str or a list of str')
    for ext in file_formats:
        file_handlers[ext] = handler


def register_handler(file_formats: Union[str, list], **kwargs) -> Callable:

    def wrap(cls):
        _register_handler(cls(**kwargs), file_formats)
        return cls

    return wrap
