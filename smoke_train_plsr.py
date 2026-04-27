import train_plsr


def main():
    train_plsr.PER_CELL_DIR.mkdir(parents=True, exist_ok=True)
    train_plsr.run_one_cell("china", "none")


if __name__ == "__main__":
    main()
