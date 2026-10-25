import argparse,json,sys
import pandas as pd
from .core import SemanticAtlas,synthesize,write_wav

from . import __version__


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        from .user_data import convert
        p = argparse.ArgumentParser(description='Convert externally collected AVAS semantic ratings')
        p.add_argument('command', choices=['convert'])
        p.add_argument('input')
        p.add_argument('--schema', required=True)
        p.add_argument('--output-dir', default='converted_ratings')
        a = p.parse_args()
        print(json.dumps(convert(a.input, a.schema, a.output_dir), indent=2))
        return
    p=argparse.ArgumentParser(description='Semantic atlas ranking or digital prototype synthesis',
        epilog='Import your own data first with: semanticavas convert INPUT --schema SCHEMA.json --output-dir DIR (see docs/USER_DATA_IMPORT.md).')
    p.add_argument('--version', action='version', version='%(prog)s '+__version__)
    sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('rank');r.add_argument('csv');r.add_argument('--target',required=True,help='JSON scale-to-target mapping');r.add_argument('--output',default='ranking.csv');r.add_argument('--weights',default=None,help='JSON scale weights');r.add_argument('--scale-ranges',default=None,help='JSON external scale widths')
    s=sub.add_parser('synthesize');s.add_argument('--f0',type=float,default=80);s.add_argument('--cutoff',type=float,default=2000);s.add_argument('--output',default='prototype.wav')
    a=p.parse_args()
    if a.command=='rank':
        target=json.loads(a.target);atlas=SemanticAtlas(pd.read_csv(a.csv),list(target));atlas.rank(target,weights=None if a.weights is None else json.loads(a.weights),scale_ranges=None if a.scale_ranges is None else json.loads(a.scale_ranges)).to_csv(a.output)
    else:write_wav(a.output,synthesize(f0=a.f0,cutoff=a.cutoff))
if __name__=='__main__':main()
