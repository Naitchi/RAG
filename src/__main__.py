from chunker import Chunker


def main():
    # TODO rajouter un argparser pour recuperer les parametres.

    chunker = Chunker(vllm_path="./vllm-0.10.1")
    chunker.get_files_and_chunks()


if __name__ == "__main__":
    main()
