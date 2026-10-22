"""Count valid raw MegaVul projects without a hard-coded platform path."""
import argparse
from collections import Counter
from pathlib import Path
from .preprocessing import iter_records,parse_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path,nargs='?',default=Path('data/raw/megavul/megavul.json'))
    args=parser.parse_args();counts=Counter();exclusions=Counter()
    for record in iter_records(args.path):
        row,reason=parse_record(record)
        if reason: exclusions[reason]+=1
        else: counts[row['project']]+=1
    print(dict(project_counts=counts,exclusions=exclusions))

if __name__=='__main__': main()
