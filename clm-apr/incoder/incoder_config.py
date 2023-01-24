InCoderInputConfig = {
    "INCODER_COMPLETE_CODEFORM_NOCOMMENT": {
        "model_id": "incoder-1B/6B",
        "input": "whole buggy function, with the bugggy line masked by <|mask:0|>",
        "patch": "code generated by the model, which will replace the entire buggy function. need extra analysis to figure out where to stop"
    },
    "INCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "incoder-1B/6B",
        "input": "whole buggy function, with the bugggy line masked by <|mask:0|>",
        "patch": "the buggy function before the buggy lines, with buggy lines start with '// buggy line:'. remove all the other commonts and empty lines in the code"
    }
}