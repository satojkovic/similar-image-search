import streamlit as st


def main():
    readme_text = st.markdown(
        get_file_content_as_string_from_local('instructions.md'))


def get_file_content_as_string_from_local(path):
    with open(path, 'r') as f:
        content = f.read()
    return content


if __name__ == '__main__':
    main()
