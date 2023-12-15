FROM public.ecr.aws/lambda/python:3.11

# Copy requirements.txt
# COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install --upgrade pip
RUN pip install pysqlite3-binary
RUN pip install python-dotenv
RUN pip3 install yfinance
RUN pip install joblib
RUN pip install scikit-learn
# RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY app ${LAMBDA_TASK_ROOT}


CMD [ "lambda_function.handler" ]

EXPOSE 8080