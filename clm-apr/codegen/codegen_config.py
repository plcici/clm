CodeGenInputConfig = {
    "CODEGEN_COMPLETE_CODEFORM_NOCOMMENT": {
        "model_id": "codegen-350M/2B/6B-multi",
        "input": "buggy function before",
        "patch": "code generated by the model, which will replace the entire buggy function. need extra analysis to figure out where to stop"
    },
    "CODEGEN_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "codegen-350M/2B/6B-multi",
        "input": "buggy function before",
        "patch": "the buggy function before the buggy lines, with buggy lines start with '// buggy line:'. remove all the other commonts and empty lines in the code"
    }
}