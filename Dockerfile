FROM public.ecr.aws/lambda/python:3.12

# Install the specified packages
RUN pip install --upgrade pip
RUN pip install pysqlite3-binary
# RUN pip install python-dotenv
RUN pip install yfinance
RUN pip install joblib
RUN pip install scikit-learn
RUN pip install boto3

# Copy function code
COPY app ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.handler" ]