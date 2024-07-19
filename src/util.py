import os


###############################################################################
# I/O
###############################################################################
def initialize_folders():
    main_folders = ["logs", "models", "result/tables"]
    for folder in main_folders:
        if not os.path.exists(f"./{folder}"):
            os.makedirs(f"./{folder}")
