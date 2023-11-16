# Docker compose for sycamore

Runs the entire Sycamore stack + Aryn demo UI.

1. Download Docker & Docker compose
   1. Either https://www.docker.com/products/docker-desktop/
   1. Or via your local package manager, e.g. `apt install docker-compose-v2`
1. Get an OpenAI API Key from https://platform.openai.com/account/api-keys
1. If you want to use textract to get good results on the sort benchmark sample data:
   1. Get an AWS account from https://repost.aws/knowledge-center/create-and-activate-aws-account
   1. Create an S3 bucket in that account, e.g. s3://_username_-textract-bucket
   1. Note: We recommend you set up lifecycle rules to delete the uploaded files automatically
1. Download the Docker compose files
   1. You may have already done this since this README is one of the files
   1. https://github.com/aryn-ai/sycamore/tree/main/docker_compose
1. Setup your environment
   1. export SYCAMORE_TEXTRACT_PREFIX=s3://_username_-textract-bucket
   1. If you are skipping textract
      1. % export ENABLE_TEXTRACT=false
   1. If you are enabling textract
      1. % aws sso login
      1. % eval "$(aws configure export-credentials --format env)"
      1. \# or any other way to setup AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and if needed AWS_SESSION_TOKEN
      1. You can verify it is working by running aws s3 ls; you should see the bucket you created for $SYCAMORE_TEXTRACT_PREFIX
1. Start docker service
   1. On MacOS or Windows, start Docker desktop
   1. On Linux if you used your local package manager it should already be started
1. Start the service
   1. In the directory with this README and the compose.yaml file
   1. % docker compose up
1. Wait until the service has finished importing a single sort benchmark pdf
   1. You will see messages like 'No changes at <date> sleeping'
1. Try out the Aryn conversational UI
   1. Visit http://localhost:3000
1. Load more files into the service
   1. % docker compose -f sort-all.yaml up
   1. You can interact with the UI while it is loading, but the data won't all be there
1. Load your files into the service
   1. TBA

# Troubleshooting

## Nothing is importing

Look at the log messages for the sycamore container: `docker compose logs sycamore`

1. If you are missing environment variables, follow the instructions above to set them up.

1. If it is constantly OOMing, increase the memory for your docker container
   1. Via Docker Desktop: Settings > Resources > scroll down > ...
   1. The minimum successfully tested resources were 4GB RAM, 4GB Swap, 64GB disk space

## Reset the configuration entirely

If the persistent data is in a confused state, you can reset it back to a blank state:

`% docker compose -f reset.yaml up`

## Reach out for help on the sycamore slack channel

https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg
