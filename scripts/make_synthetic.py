"""Generate Massive-STEPS-format synthetic data + a tiny FSQ taxonomy CSV so the
whole pipeline can be smoke-tested offline."""
import argparse, os
import numpy as np, pandas as pd

# tiny FSQ-style taxonomy: id -> "Activity > Primary > Secondary"
FSQ = [
    ("f01","Coffee Shop","Dining and Drinking > Cafe > Coffee Shop"),
    ("f02","Restaurant","Dining and Drinking > Restaurant"),
    ("f03","Bar","Dining and Drinking > Bar"),
    ("f04","Office","Business and Professional Services > Office"),
    ("f05","Gym","Health and Medicine > Gym / Fitness Center"),
    ("f06","Park","Landmarks and Outdoors > Park"),
    ("f07","Train Station","Travel and Transportation > Transport Hub > Train Station"),
    ("f08","Grocery Store","Retail > Grocery Store"),
    ("f09","Mall","Retail > Shopping Mall"),
    ("f10","Home","Community and Government > Housing > Home"),
]
CATS = [(c[0], c[1]) for c in FSQ]
TOURS = [["f10","f04","f10"],["f10","f01","f04","f10"],["f10","f04","f02","f10"],
         ["f10","f05","f10","f03","f10"],["f10","f07","f04","f07","f10"],
         ["f10","f08","f10"],["f10","f01","f06","f02","f10"]]

def gen(city, n_users, days, seed):
    rng = np.random.RandomState(seed)
    clat, clon = rng.uniform(-40,40), rng.uniform(-100,100)
    pois = {c:[(f"{n} {i}", clat+rng.randn()*.05, clon+rng.randn()*.05,
               (hash(city)%9000)+ci*100+i) for i in range(40)]
            for ci,(c,n) in enumerate(CATS)}
    rows=[]; tid=0
    for u in range(n_users):
        pref = rng.dirichlet(np.ones(len(TOURS)))
        for _ in range(days):
            day0 = pd.Timestamp("2018-01-01")+pd.Timedelta(days=int(rng.randint(0,300)))
            tour = TOURS[rng.choice(len(TOURS), p=pref)]
            t = day0+pd.Timedelta(hours=7)
            for step,cid in enumerate(tour):
                name,la,lo,vid = pois[cid][rng.randint(40)]
                cname = dict(CATS)[cid]
                rows.append(dict(trail_id=tid,user_id=u,venue_id=int(vid),
                    latitude=la,longitude=lo,name=name,address="n/a",
                    venue_category=cname,venue_category_id=cid,venue_category_id_code=cid,
                    venue_city=city,venue_city_latitude=clat,venue_city_longitude=clon,
                    venue_country="SYN",
                    timestamp=(t+pd.Timedelta(hours=2*step)).strftime("%Y-%m-%d %H:%M:%S")))
            tid+=1
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="data")
    ap.add_argument("--cities",nargs="+",default=["src_a","src_b","tgt_c"])
    ap.add_argument("--n_users",type=int,default=80); ap.add_argument("--days",type=int,default=20)
    ap.add_argument("--fsq_out",default="personalization-apis-movement-sdk-categories.csv")
    a=ap.parse_args()
    pd.DataFrame([{"Category ID":c[0],"Category Name":c[1],"Category Label":c[2]} for c in FSQ]
                ).to_csv(a.fsq_out,index=False)
    for i,c in enumerate(a.cities):
        df=gen(c,a.n_users,a.days,7+i)
        tr=df.trail_id.unique(); rng=np.random.RandomState(0); rng.shuffle(tr)
        n=len(tr); parts={"train":tr[:int(.7*n)],"validation":tr[int(.7*n):int(.8*n)],"test":tr[int(.8*n):]}
        d=os.path.join(a.out,c); os.makedirs(d,exist_ok=True)
        for sp,ids in parts.items():
            df[df.trail_id.isin(ids)].to_csv(os.path.join(d,f"{c}_checkins_{sp}.csv"),index=False)
        print(f"{c}: {len(df)} check-ins, {n} trails")
    print("wrote FSQ taxonomy ->", a.fsq_out)

if __name__=="__main__": main()
