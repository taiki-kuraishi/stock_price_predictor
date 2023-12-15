build
```sh
docker build -t my-lambda-image .
```

run
```sh
docker run --rm -p 9000:8080 --name aws-lamda  my-lambda-image
```

rm conteiner
```sh
docker rm aws-lamda
```