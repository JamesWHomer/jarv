def prompt(text):
    return input(text).strip()


def main():
    name = ""
    visited_reports = False

    while True:
        print()
        print("Jarv Interactive Menu Test")
        print("1. Profile")
        print("2. Reports")
        print("3. Exit")
        choice = prompt("Choose an option: ")

        if choice == "1":
            name = prompt("Enter profile name: ")
            print(f"Profile saved for {name}.")
        elif choice == "2":
            print()
            print("Reports")
            print("1. Daily")
            print("2. Weekly")
            report_choice = prompt("Choose a report: ")
            if report_choice == "2":
                visited_reports = True
                who = name or "unknown"
                print(f"Weekly report opened for {who}.")
            else:
                print("Daily report opened.")
        elif choice == "3":
            if name and visited_reports:
                print(f"Objective complete: {name} opened the weekly report.")
                return
            print("Objective incomplete. Set a profile name and open the weekly report before exiting.")
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
