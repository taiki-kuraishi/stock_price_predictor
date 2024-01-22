build
```sh
docker build -t aws-lambda-spp .
```

run
```sh
docker run --rm -p 9000:8080 --name aws-lambda-spp  my-lambda-image
```

rm conteiner
```sh
docker rm aws-lambda-spp
```