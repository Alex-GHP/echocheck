def main():
    with open("/data/data_center.json", "r") as f:
        print(f.read()[:4000])


if __name__ == "__main__":
    main()
