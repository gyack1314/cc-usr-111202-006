"""查看原始研究资料的已有摘要与选择性披露敏感性分析。"""
import argparse
import json
from .io import load
from .domain import summarize
from .selection import analyze

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect','sensitivity']);p.add_argument('file');a=p.parse_args()
    value,digest=load(a.file)
    if a.command=='inspect':
        out={'source_sha256':digest,'summary':summarize(value)}
    else:
        out={'source_sha256':digest,**analyze(value)}
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
