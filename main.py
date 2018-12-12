import argparse
import datetime
from pathlib import Path

from allennlp.commands import main, Subcommand
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.models import Model


class MyTrain(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(name, description=description, help='Train a model')

        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        subparser.add_argument('-s', '--serialization-dir',
                               required=False,
                               default="",
                               type=str,
                               help='directory in which to save the model and its logs')

        subparser.add_argument('-r', '--recover',
                               action='store_true',
                               default=False,
                               help='recover training from the state in serialization_dir')

        subparser.add_argument('-f', '--force',
                               action='store_true',
                               required=False,
                               help='overwrite the output directory if it exists')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')


        subparser.add_argument('-e', '--ext-vars',
                               type=str,
                               default=None,
                               help='Used to provide ext variable to jsonnet')

        subparser.add_argument('--fp16',
                               action='store_true',
                               required=False,
                               help='use fp 16 training')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """

    start_time = datetime.datetime.now().strftime('%b-%d_%H-%M')

    if args.serialization_dir:
        serialization_dir = args.serialization_dir
    else:
        path = Path(args.param_path.replace("configs/", "results/")).resolve()
        serialization_dir = path.with_name(path.stem) / start_time


    train_model_from_file(args.param_path,
                          serialization_dir,
                          args.overrides,
                          args.file_friendly_logging,
                          args.recover,
                          args.force,
                          args.ext_vars)

def train_model_from_file(parameter_filename: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          recover: bool = False,
                          force: bool = False,
                          ext_vars=None) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    param_path : ``str``
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides, ext_vars=ext_vars)
    return train_model(params, serialization_dir, file_friendly_logging, recover, force)


if __name__ == "__main__":
    import_submodules("qa")
    import_submodules("modules")
    main(prog="ReadingZoo",subcommand_overrides={"train": MyTrain()})