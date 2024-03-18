# Spaceship_Titanic_MLOps_Project

Predict which passengers are transported to an alternate dimension

We will be concentraining on MLOps part only, topics like feature engineering, trying alternate models are given less importance.

Difference between continours delivery and deployment 
Continous Integration: code commits pushes , reviews 
Continuous Delivery: ensures that code is always in a deployable state,  with automated testing
Continuous Deployment : deployment to production.


## Note Use the same code base - 
- change the db settings alone and test locally , 
- Added torch code and flask code additionally 
- test it locally then deploy to AWS
Initate git and set git url 
initiate docker . dont run in same environment 

docker build -t spaceship_pro_1 .
600 - 1000 seconds to build 
docker run -p 8085:8085 spaceship_pro_1 
docker exec -it 15e539e58a68  bash
so its able to run properly in a docker env 
url/train - even if you change the database model will be generated with latest data

now push to git in separate repo

it would cost you around 30 ruppes u need to pay at month end to AWS if you are going to proceed further
now go to aws 


## AWS
1. Create a iam user with following policies -> Attach policis directly

    a. AmazonEC2ContainerRegistryFullAccess

    b. AmazonEC2FullAccess

2. Create a new keyvalue pair under security credentials and save the file

3. Now create ECR repo and store the URL

    306093656765.dkr.ecr.us-east-1.amazonaws.com/spaceship

### Go to EC2
- create keyvalue pair if u want to access through putty.
t2.xlarge - select
Allow HTTP and HTTPs traffic

### Install docker

    sudo apt-get update -y

    sudo apt-get upgrade

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker

## Now Goto Github
- Go to Actions -> Runners -> new self hosted runner
- Execute all commands for linux machine as we selected it

- While entering name of runner give "self-hosted" runner

## Github secrets

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = only till .com

ECR_REPOSITORY_NAME = nameof ECR repo

IN EC2:
change inbound rules for 8085 port
access http://54.204.114.208:8085/

http://localhost:8081/models/my_model
http://localhost:8080/predictions/my_model

**NOTE : Terminate the instance, delete ECR repo, delete IAM user**

References:
- Krish Naik Videos
- https://github.com/dimitreOliveira/torchserve_od_example/tree/main