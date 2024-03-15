from huggingface_hub import Repository


if __name__ == '__main__':
    repo = Repository(
        local_dir=".",
        clone_from="lfrachon/lyrics-dreamer",

        repo_type="model",
    )
    # repo.git_pull()
    repo.git_add("checkpoints")
    repo.git_commit("Update checkpoints")
    repo.git_push("main")
