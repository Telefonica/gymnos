#
#
#   Gymnos CLI
#
#

import argparse

from . import deploy, train, predict, serve


def build_parser():
    parser = argparse.ArgumentParser(description="Gymnos tool")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train gymnos trainer using a JSON file")
    train.add_arguments(train_parser)

    predict_parser = subparsers.add_parser("predict", help="Predict samples using a saved trainer")
    predict.add_arguments(predict_parser)

    serve_parser = subparsers.add_parser("serve", help="Serve predictions using a saved trainer")
    serve.add_arguments(serve_parser)

    deploy_parser = subparsers.add_parser("deploy", help="Deploy a saved trainer to SOFIA")
    deploy.add_arguments(deploy_parser)

    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()

    if args.command == "train":
        module = train
    elif args.command == "predict":
        module = predict
    elif args.command == "serve":
        module = serve
    elif args.command == "deploy":
        module = deploy
    else:
        return parser.print_help()

    module.run_command(args)


if __name__ == "__main__":
    main()
