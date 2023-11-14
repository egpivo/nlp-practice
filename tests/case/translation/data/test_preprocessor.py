import os

from nlp_practice.case.translation.data.preprocessor import Preprocessor


def test_preprocessor_process():
    file_contents = ["Go.\tVa !", "Hello.\tSalut !", "How are you?\tComment ça va ?"]
    test_base_path = "."
    test_file_path = os.path.join(test_base_path, "eng-fra.txt")
    with open(test_file_path, "w", encoding="utf-8") as test_file:
        test_file.write("\n".join(file_contents))

    input_language, output_language, valid_pairs = Preprocessor(
        base_path=test_base_path
    ).process()

    # Check if the returned values are not None
    assert input_language is not None
    assert output_language is not None
    assert valid_pairs is not None
    assert len(valid_pairs) == 0
    assert input_language.num_words == 2
    assert output_language.num_words == 2
