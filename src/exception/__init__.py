import os
import sys


class CustomException(Exception):

    def __init__(
        self,
        error_message: Exception,
        error_detail=sys,
    ):

        self.error_message = self.get_detailed_error_message(
            error_message,
            error_detail,
        )

        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(
        error: Exception,
        error_detail,
    ) -> str:

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:

            file_name = os.path.basename(
                exc_tb.tb_frame.f_code.co_filename
            )

            line_number = exc_tb.tb_lineno

        else:

            file_name = "Unknown"

            line_number = -1

        return (
            f"Error occurred in [{file_name}] "
            f"at line [{line_number}] : {str(error)}"
        )

    def __str__(self):

        return self.error_message