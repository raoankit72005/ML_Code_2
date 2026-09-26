"""Print a reviewable Training Job request; only --submit starts paid compute."""
import argparse
import json
import re
from datetime import datetime, timezone


def request(args):
    if not args.role.startswith('arn:aws:iam::'):raise ValueError('Provide a SageMaker execution role ARN')
    if not args.output.startswith('s3://') or not args.input.startswith('s3://'):raise ValueError('Input/output must be S3 URIs')
    if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?',args.name):raise ValueError('Invalid job name')
    profile=getattr(args,'profile','g6e')
    if profile not in ('g6e','g5'):raise ValueError('Profile must be g6e or g5')
    if profile=='g5' and ':sm86-' not in args.image:
        raise ValueError('G5 requires an SM86 image built with CUDA_ARCH=86')
    if profile=='g6e' and ':sm86-' in args.image:
        raise ValueError('SM86 image cannot be used with G6e')
    prefix=args.output.rstrip('/')+'/'+args.name
    return dict(TrainingJobName=args.name,RoleArn=args.role,
        AlgorithmSpecification=dict(TrainingImage=args.image,TrainingInputMode='File',
            MetricDefinitions=[dict(Name='encoder:loss',Regex=r'"loss": ([0-9.eE+-]+)'),
                               dict(Name='validation:macro_f05',Regex=r'"macro_f05": ([0-9.eE+-]+)')]),
        InputDataConfig=[dict(ChannelName='dataset',DataSource=dict(S3DataSource=dict(
            S3DataType='S3Prefix',S3Uri=args.input,S3DataDistributionType='FullyReplicated')))],
        OutputDataConfig=dict(S3OutputPath=prefix+'/artifacts'),
        CheckpointConfig=dict(S3Uri=prefix+'/checkpoints',LocalPath='/opt/ml/checkpoints'),
        ResourceConfig=dict(InstanceType='ml.g5.xlarge' if profile=='g5' else 'ml.g6e.12xlarge',InstanceCount=1,VolumeSizeInGB=args.volume_gb),
        StoppingCondition=dict(MaxRuntimeInSeconds=args.max_hours*3600),
        EnableManagedSpotTraining=False,EnableNetworkIsolation=False,
        Environment=dict(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2' if profile=='g5' else '8',MKL_NUM_THREADS='2' if profile=='g5' else '8',ER_PROFILE=profile))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',required=True);p.add_argument('--image',required=True,help='Your built CUDA image URI in ECR')
    p.add_argument('--input',default='s3://ml-hack1/ML_dataset.zip');p.add_argument('--output',required=True)
    p.add_argument('--region',default='us-east-1');p.add_argument('--name',default='ml-code-2-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    p.add_argument('--profile',choices=('g6e','g5'),default='g6e')
    p.add_argument('--volume-gb',type=int,default=1000);p.add_argument('--max-hours',type=int,default=24)
    p.add_argument('--submit',action='store_true');a=p.parse_args()
    if a.volume_gb<1 or a.max_hours<1:p.error('Volume and max-hours must be positive')
    payload=request(a);print(json.dumps(payload,indent=2))
    if a.submit:
        import boto3
        print(json.dumps(boto3.client('sagemaker',region_name=a.region).create_training_job(**payload),default=str,indent=2))
    else:print('DRY RUN. Add --submit to start the Training Job (incurs AWS charges).')

if __name__=='__main__':main()
