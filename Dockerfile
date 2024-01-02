FROM public.ecr.aws/lambda/python:3.12

RUN pip install --upgrade pip

COPY ./requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt 
# RUN pip install pysqlite3-binary

# Copy function code
COPY app ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.lambda_handler" ]