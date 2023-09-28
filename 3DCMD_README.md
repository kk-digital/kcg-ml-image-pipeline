# 3scmd tool with our minio server

### 3scmd:
Download 3dcmd from github:

run:
```
    git clone https://github.com/s3tools/s3cmd
```
### Configure 3scmd
run:

```
    cd ./s3cmd
```

```
    python s3cmd --configure
```
   
Configuration instructions:
``` 
Set the following configuration

    - Access key: Your access key
    - Secrey Key : Your secret key
    - S3 Endpoint: 123.176.98.90:9000
    - DNS-style bucket+hostname:port template for accessing a bucket: %(bucket)123.176.98.90:9000
    - Encryption password:
    - Path to GPG program: None
    - Use HTTPS protocol: False
    - HTTP Proxy server name:
    - HTTP Proxy server port: 0
    
```

### Running 3dcmd commands:

The most common command to use is the sync command, Here is an example

Windows:
``` 
python s3cmd sync s3://datasets .\datasets\
``` 

Linux:
``` 
python s3cmd sync s3://datasets ./datasets/
``` 

##### Other commands:

To make a bucket
``` 
python s3cmd mb s3://mybucket
``` 

To copy an object to bucket
``` 
python s3cmd mb s3://mybucket
``` 

To make a bucket
``` 
python s3cmd put newfile s3://testbucket
``` 

To copy an object to local system
``` 
python s3cmd get s3://testbucket/newfile
``` 

To sync local file/directory to a bucket
``` 
python s3cmd sync newdemo s3://testbucket
``` 

To sync bucket or object with local filesystem
``` 
python s3cmd sync s3://testbucket otherlocalbucket
``` 

To list buckets
``` 
python s3cmd ls s3://
``` 

To list contents inside bucket
``` 
python s3cmd ls s3://testbucket/
``` 

To delete an object from bucket
``` 
python s3cmd del s3://testbucket/newfile
``` 

To delete a bucket
``` 
python rb s3://mybucket
``` 

