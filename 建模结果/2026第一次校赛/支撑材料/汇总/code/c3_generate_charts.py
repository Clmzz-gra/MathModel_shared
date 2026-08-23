"""
c3_generate_charts.py — 批量生成18张图表 + 重出2张已有图表
使用 chart_utils 统一接口: setup_mpl + resolve_dirs + save_figure(review=True)
"""
import sys, os
sys.path.insert(0, 'E:/MathModel-school-competition')
from chart_utils import setup_mpl, save_figure, resolve_dirs

setup_mpl()
FIG_DIR, CHART_DIR = resolve_dirs(__file__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 数据路径
DATA_DIR = os.path.join(os.path.dirname(FIG_DIR), 'data')
df = pd.read_pickle(os.path.join(DATA_DIR, 'combined-clean.pkl'))
q3df = pd.read_pickle(os.path.join(DATA_DIR, 'q3-judge-scores.pkl'))
TOPIC_LABELS = ['A题','B题','C题','D题','E题']
TOPICS = ['A','B','C','D','E']
award_map = {'一等奖':3,'二等奖':2,'三等奖':1}

# 全局计算一次 Q1 所需中间数据 — 按评委ID做z-score
judge_stats = {}
for j in range(1, 5):
    for _, row in df.iterrows():
        jid = row[f'评委{j}']
        score = row[f'打分{j}']
        if jid not in judge_stats:
            judge_stats[jid] = []
        judge_stats[jid].append(score)
judge_mu_sigma = {jid: (np.mean(scores), np.std(scores)) for jid, scores in judge_stats.items()}

records = []
for idx, row in df.iterrows():
    z_scores = []
    for j in range(1, 5):
        jid = row[f'评委{j}']
        mu, sigma = judge_mu_sigma[jid]
        raw = row[f'打分{j}']
        z = (raw - mu) / sigma if sigma > 0 else 0
        z_scores.append(z)
    z_mean = np.mean(z_scores)
    award_val = award_map.get(row['成绩'], 0)
    records.append({'题目': row['题目'], '阅卷号': row['阅卷号'],
                   'z_mean': z_mean, 'award_val': award_val,
                   'has_award': pd.notna(row['成绩']), 'is_first': row['成绩'] == '一等奖'})
q1df = pd.DataFrame(records)

# ============================================================
# Q1 图表 (4张)
# ============================================================

# --- 计算各题 Spearman ρ ---
rhos = {}
for t in TOPICS:
    sub = q1df[q1df['题目']==t]
    rhos[t],_ = spearmanr(sub['z_mean'],sub['award_val'])
rhos['全题'],_ = spearmanr(q1df['z_mean'],q1df['award_val'])

# --- Q1-1: ρ 柱状图 + 效应量 ---
fig,ax=plt.subplots(figsize=(7,4))
vals=[rhos[t] for t in TOPICS]
bars=ax.bar(TOPIC_LABELS,vals,color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'])
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{v:.3f}',ha='center',fontsize=10,fontweight='bold')
ax.axhline(y=0.5,color='gray',linestyle='--',alpha=0.5,label='强相关阈值(0.5)')
ax.axhline(y=0.3,color='gray',linestyle=':',alpha=0.5,label='中等相关阈值(0.3)')
ax.set_ylim(0,1); ax.set_ylabel('Spearman ρ'); ax.set_title('各题网评效度')
ax.legend(fontsize=8)
save_figure(fig,'q1-rho-bars',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题1：各题Spearman ρ+效应量阈值',review=True)

# --- Q1-3: ROC 曲线 ---
from sklearn.metrics import roc_curve, auc
fig,ax=plt.subplots(figsize=(6,5))
for t,c in zip(TOPICS,['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']):
    sub=q1df[q1df['题目']==t]
    if sub['is_first'].sum()>3:
        fpr,tpr,_=roc_curve(sub['is_first'],sub['z_mean'])
        a=auc(fpr,tpr); ax.plot(fpr,tpr,color=c,label=f'{t} (AUC={a:.3f})')
ax.plot([0,1],[0,1],'k--',alpha=0.3,label='随机')
ax.set_xlabel('假阳性率'); ax.set_ylabel('真阳性率')
ax.set_title('ROC: 网评标准分预测一等奖'); ax.legend(fontsize=8)
save_figure(fig,'q1-roc-curves',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题1：五题ROC曲线含AUC',review=True)

# --- Q1-4: 筛选命中率 ---
fig,ax=plt.subplots(figsize=(7,4))
hit_rates=[]; fn_rates=[]
for t in TOPICS:
    sub=q1df[q1df['题目']==t].sort_values('z_mean',ascending=False)
    n=int(len(sub)*0.55); top=sub.head(n)
    hr=top['has_award'].sum()/len(top)
    fn=sub.tail(len(sub)-n)['has_award'].sum()/sub['has_award'].sum()
    hit_rates.append(hr); fn_rates.append(fn)
x=np.arange(5); w=0.35
ax.bar(x-w/2,hit_rates,w,label='命中率',color='#2ca02c')
ax.bar(x+w/2,fn_rates,w,label='假阴性率',color='#d62728')
ax.set_xticks(x); ax.set_xticklabels(TOPIC_LABELS)
for i in range(5):
    ax.text(i-w/2,hit_rates[i]+0.01,f'{hit_rates[i]:.1%}',ha='center',fontsize=9)
    ax.text(i+w/2,fn_rates[i]+0.01,f'{fn_rates[i]:.1%}',ha='center',fontsize=9)
ax.set_ylabel('比例'); ax.set_title('网评筛选有效性'); ax.legend(loc='upper left')
save_figure(fig,'q1-hit-rate',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题1：网评筛选命中率+假阴性率',review=True)

# --- Q1-5: 散点图矩阵 ---
fig,axes=plt.subplots(2,3,figsize=(12,8))
axes=axes.flatten()
for idx,t in enumerate(TOPICS):
    sub=q1df[q1df['题目']==t]
    ax=axes[idx]
    ax.scatter(sub[~sub['has_award']]['z_mean'],sub[~sub['has_award']]['award_val'],
              c='gray',alpha=0.3,s=10,label='淘汰')
    ax.scatter(sub[sub['has_award']]['z_mean'],sub[sub['has_award']]['award_val'],
              c='#d62728',alpha=0.5,s=10,label='获奖')
    ax.set_title(f'{t} (ρ={rhos[t]:.3f})',fontsize=10)
    ax.set_xlabel('网评标准分'); ax.set_ylabel('奖项等级')
axes[5].axis('off')
save_figure(fig,'q1-scatter-matrix',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题1：网评标准分vs奖项散点图矩阵',review=True)

# ============================================================
# Q2+Q3 图表 (9张)
# ============================================================

# --- Q2-1: 指标体系层次结构图 ---
fig,ax=plt.subplots(figsize=(8,5))
ax.set_xlim(0,10); ax.set_ylim(0,7); ax.axis('off')
ax.text(5,6.5,'评委基本素质',ha='center',fontsize=14,fontweight='bold',
        bbox=dict(boxstyle='round',facecolor='#1f77b4',alpha=0.2))
for i,(dim,stake,x) in enumerate([('信度\nReliability','其他评委\n可复现',1.5),
    ('效度\nValidity','参赛者\n看得准',3.5),('公平性\nFairness','组织者\n无偏见',5.5),
    ('区分力\nDiscrimination','学术标准\n有区分',7.5)]):
    ax.text(x,4.5,dim,ha='center',fontsize=10,fontweight='bold',
            bbox=dict(boxstyle='round',facecolor='#ff7f0e',alpha=0.2))
    ax.text(x,3.5,stake,ha='center',fontsize=8,color='gray')
    ax.plot([5,x],[6.2,4.7],'gray',alpha=0.5)
for i,(metric,x) in enumerate([('成对ICC均值',1.5),('Spearman ρ',3.5),
    ('|偏差z|',5.5),('标准差/CV',7.5)]):
    ax.text(x,2.2,metric,ha='center',fontsize=9,color='#333333',
            bbox=dict(boxstyle='round',facecolor='#2ca02c',alpha=0.15))
    ax.plot([x,x],[4.3,2.5],'gray',alpha=0.5)
ax.set_title('Q2: 评委基本素质指标体系 (四维度·利益相关者视角)',fontsize=12)
save_figure(fig,'q2-indicator-architecture',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题2：四维度指标体系层次结构图',review=True)

# --- Q3-1: 熵权柱状图 ---
entropy_weights = {'信度':{'A':0.153,'B':0.203,'C':0.199,'D':0.203,'E':0.153},
    '效度':{'A':0.224,'B':0.203,'C':0.216,'D':0.183,'E':0.383},
    '公平性':{'A':0.240,'B':0.152,'C':0.222,'D':0.177,'E':0.159},
    '区分力':{'A':0.382,'B':0.442,'C':0.363,'D':0.437,'E':0.304}}
fig,ax=plt.subplots(figsize=(8,5))
x=np.arange(5); w=0.2; colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728']
for i,(dim,c) in enumerate(zip(['信度','效度','公平性','区分力'],colors)):
    vals=[entropy_weights[dim][t] for t in TOPICS]
    ax.bar(x+i*w,vals,w,label=dim,color=c,alpha=0.8)
ax.set_xticks(x+w*1.5); ax.set_xticklabels(TOPIC_LABELS)
ax.set_ylabel('熵权'); ax.set_title('五题四维度熵权分布'); ax.legend()
save_figure(fig,'q3-entropy-weights',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题3：五题熵权法权重分布对比',review=True)

# --- Q3-2: TOPSIS 直方图 ---
fig,axes=plt.subplots(2,3,figsize=(12,7)); axes=axes.flatten()
for idx,t in enumerate(TOPICS):
    sub=q3df[q3df['题目']==t]
    axes[idx].hist(sub['TOPSIS得分'],bins=12,color='#1f77b4',alpha=0.7,edgecolor='white')
    axes[idx].axvline(sub['TOPSIS得分'].mean(),color='#d62728',linestyle='--',label=f'均值={sub["TOPSIS得分"].mean():.3f}')
    axes[idx].set_title(f'{t} ({len(sub)}位)'); axes[idx].legend(fontsize=8)
axes[5].axis('off')
fig.suptitle('各题评委TOPSIS得分分布',fontsize=13)
plt.tight_layout()
save_figure(fig,'q3-topsis-hist',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题3：五题TOPSIS得分直方图',review=True)

# --- Q3-3: 雷达图 ---
dim_names=['信度','效度','公平性','区分力']
dim_cols=['信度','效度','公平性_raw','区分力']  # 公平性_raw=|z_bias|, 越大越差, 后续取反
fig,axes=plt.subplots(2,3,figsize=(12,8),subplot_kw=dict(projection='polar'))
axes=axes.flatten()
for idx,t in enumerate(TOPICS):
    sub=q3df[q3df['题目']==t]; ax=axes[idx]
    for layer_name,color,ls in [('优秀','#2ca02c','-'),('良好','#1f77b4','-.'),('合格','#ff7f0e','--'),('需关注','#d62728',':')]:
        layer=sub[sub['分层']==layer_name]
        if len(layer)>0:
            vals=[layer[dim_cols].mean()[dc].item() if dc in layer.columns else 0 for dc in dim_cols]
            max_vals=sub[dim_cols].max().values
            norm_vals=[v/m if m>0 else 0 for v,m in zip(vals,max_vals)]
            norm_vals[2] = 1 - norm_vals[2]  # 公平性: raw越大越差 → 取反使越大越好
            angles=np.linspace(0,2*np.pi,len(dim_names),endpoint=False).tolist()
            norm_vals.append(norm_vals[0]); angles.append(angles[0])
            ax.plot(angles,norm_vals,color=color,linestyle=ls,linewidth=1.5,label=layer_name)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(dim_names,fontsize=8)
    ax.set_title(t,fontsize=10)
axes[5].axis('off')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower right',fontsize=8)
fig.suptitle('各分层评委四维度雷达图',fontsize=13)
save_figure(fig,'q3-radar',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题3：各分层评委四维度雷达图',review=True)

# --- Q3-4: 肘部+Silhouette ---
print('Generating Q3-4 elbow/silhouette plots...')
fig,axes=plt.subplots(2,3,figsize=(14,8)); axes=axes.flatten()
for idx,t in enumerate(TOPICS):
    sub=q3df[q3df['题目']==t]
    # 复现聚类扫描
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X=sub[['信度','效度','公平性','区分力']].values
    X_norm=(X-X.min(0))/(X.max(0)-X.min(0)+1e-6)
    S_score=np.column_stack([sub['TOPSIS得分'].values,X_norm])
    ks=range(3,min(6,len(sub)))  # K≥3业务约束
    inertias=[KMeans(n_clusters=k,init='k-means++',random_state=42,n_init=10).fit(S_score).inertia_ for k in ks]
    sils=[silhouette_score(S_score,KMeans(n_clusters=k,init='k-means++',random_state=42,n_init=10).fit_predict(S_score)) for k in ks]
    ax1=axes[idx]; ax2=ax1.twinx()
    ax1.plot(list(ks),inertias,'bo-',label='SSE'); ax1.set_ylabel('SSE',color='blue')
    ax2.plot(list(ks),sils,'ro-',label='Silhouette'); ax2.set_ylabel('Silhouette',color='red')
    ax1.set_title(f'{t}'); ax1.set_xlabel('K')
axes[5].axis('off')
fig.suptitle('肘部法则+Silhouette',fontsize=13)
plt.tight_layout()
save_figure(fig,'q3-elbow-silhouette',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题3：K-means肘部法则+Silhouette系数',review=True)

# --- Q3-5: 聚类散点图 ---
fig,axes=plt.subplots(2,3,figsize=(14,9)); axes=axes.flatten()
for idx,t in enumerate(TOPICS):
    sub=q3df[q3df['题目']==t]; ax=axes[idx]
    for layer_name,color,marker in [('优秀','#2ca02c','o'),('良好','#1f77b4','s'),
        ('合格','#ff7f0e','^'),('需关注','#d62728','x'),('待改进','#9467bd','D')]:
        layer=sub[sub['分层']==layer_name]
        if len(layer)>0:
            ax.scatter(layer['信度'],layer['区分力'],c=color,marker=marker,s=40,alpha=0.7,label=f'{layer_name}({len(layer)})')
    ax.set_xlabel('信度'); ax.set_ylabel('区分力'); ax.set_title(t);
    ax.legend(fontsize=7, edgecolor='black', framealpha=0.9)
axes[5].axis('off')
fig.suptitle('评委聚类: 信度×区分力 (按分层着色)',fontsize=13)
save_figure(fig,'q3-cluster-scatter',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题3：评委聚类散点图(信度×区分力)',review=True)

# --- Q3-6: 排名棒棒糖图 ---
fig,axes=plt.subplots(2,3,figsize=(14,9)); axes=axes.flatten()
for idx,t in enumerate(TOPICS):
    sub=q3df[q3df['题目']==t].sort_values('TOPSIS得分'); ax=axes[idx]
    topn=min(8,len(sub))
    scores=sub['TOPSIS得分'].values[-topn:]
    names=[f'{sub.iloc[i]["评委ID"]}({sub.iloc[i]["分层"]})' for i in range(len(sub)-topn,len(sub))]
    colors=['#2ca02c' if '优秀' in n else '#ff7f0e' for n in names]
    ax.hlines(range(topn),0,scores,colors=colors,linewidth=2)
    ax.scatter(scores,range(topn),c=colors,s=50,zorder=5)
    ax.set_yticks(range(topn)); ax.set_yticklabels(names,fontsize=8)
    ax.set_xlabel('TOPSIS'); ax.set_title(f'{t} TOP{topn}')
axes[5].axis('off')
fig.suptitle('各题TOPSIS得分TOP评委',fontsize=13)
plt.tight_layout()
save_figure(fig,'q3-ranking-lollipop',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题3：各题TOP评委棒棒糖图',review=True)

# ============================================================
# Q4 图表 (3张)
# ============================================================

# --- Q4-1: 箱线图 ---
fig,ax=plt.subplots(figsize=(7,5))
data=[q3df[q3df['题目']==t]['TOPSIS得分'].values for t in TOPICS]
bp=ax.boxplot(data,labels=TOPIC_LABELS,patch_artist=True)
for patch,color in zip(bp['boxes'],['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']):
    patch.set_facecolor(color); patch.set_alpha(0.4)
ax.set_ylabel('TOPSIS得分'); ax.set_title('五题评委TOPSIS得分对比 (KW p=0.076, η²=0.023, ns)')
save_figure(fig,'q4-topsis-box',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题4：五题TOPSIS得分箱线图含KW+Dunn结果',review=True)

# --- Q4-2: Dunn矩阵 ---
dunn_pvals={'A':{'A':1,'B':1,'C':1,'D':1,'E':0.632},
    'B':{'A':1,'B':1,'C':1,'D':1,'E':0.414},
    'C':{'A':1,'B':1,'C':1,'D':1,'E':0.105},
    'D':{'A':1,'B':1,'C':1,'D':1,'E':0.363},
    'E':{'A':0.632,'B':0.414,'C':0.105,'D':0.363,'E':1}}
mat=np.array([[dunn_pvals[r][c] for c in TOPICS] for r in TOPICS])
fig,ax=plt.subplots(figsize=(6,5))
# -log10 transform
mat_log=-np.log10(np.clip(mat,1e-10,1))
im=ax.imshow(mat_log,cmap='YlOrRd',vmin=0,vmax=5)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(TOPIC_LABELS); ax.set_yticklabels(TOPIC_LABELS)
for i in range(5):
    for j in range(5):
        p=mat[i,j]; sig='***' if p<0.001 else('**' if p<0.01 else('*' if p<0.05 else'ns'))
        ax.text(j,i,f'{p:.3f}{"\n"+sig}',ha='center',va='center',fontsize=8)
ax.set_title('Dunn事后检验 p_adj 矩阵 (Bonferroni)')
plt.colorbar(im,ax=ax,label='-log10(p)')
save_figure(fig,'q4-dunn-matrix',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题4：Dunn事后检验显著性矩阵',review=True)

# --- Q4-3: 四维度分题箱线图 ---
fig,axes=plt.subplots(2,2,figsize=(12,9))
dim_pairs=[('信度','Reliability'),('效度','Validity'),('公平性_raw','Fairness'),('区分力','Discrimination')]
for (col,label),ax in zip(dim_pairs,axes.flatten()):
    data_d=[q3df[q3df['题目']==t][col].dropna().values for t in TOPICS]
    bp=ax.boxplot(data_d,labels=TOPIC_LABELS,patch_artist=True)
    for patch in bp['boxes']: patch.set_facecolor('#1f77b4'); patch.set_alpha(0.3)
    ax.set_title(label)
fig.suptitle('四维度分题对比 (信度+效度+区分力 p<0.001; 公平性 p=0.59)',fontsize=13)
plt.tight_layout()
save_figure(fig,'q4-dimensions-box',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题4：四维度分题箱线图对比',review=True)

# ============================================================
# Q5 图表 (3张)
# ============================================================

# --- Q5-1: 权重敏感性（已修正, 由 gen_q5_alpha_charts.py 生成, 此处保留旧图表生成以便向后兼容）---
# 注: 旧值基于跨题聚合 spearmanr, α=0 ρ=0.660 是错误的.
# 修正后使用分题加权平均, α=0 ρ=1.000, 图表由 gen_q5_alpha_charts.py 生成.
# 以下代码保留但注释掉, 避免覆盖修正后的图.
"""
alphas=[0,0.1,0.2,0.25,0.3,0.4,0.5]
rhos_alpha=[0.660,0.668,0.664,0.658,0.652,0.636,0.613]
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(alphas,rhos_alpha,'bo-',linewidth=2,markersize=8)
ax.axvline(x=0.25,color='#d62728',linestyle='--',alpha=0.5,label='当前权重α=0.25')
ax.annotate(f'当前: ρ=0.658',xy=(0.25,0.658),xytext=(0.3,0.67),
            arrowprops=dict(arrowstyle='->',color='#d62728'),fontsize=9)
ax.set_xlabel('网评权重 α'); ax.set_ylabel('Spearman ρ')
ax.set_title('网评权重敏感性分析'); ax.legend()
save_figure(fig,'q5-weight-sensitivity',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题5：网评权重敏感性曲线',review=True)
"""

# --- Q5-2: 加权对比 ---
rho_eq=[0.513,0.511,0.303,0.661,0.574]
rho_w=[0.519,0.518,0.320,0.673,0.603]
fig,ax=plt.subplots(figsize=(7,4))
x=np.arange(5); w=0.35
bars1=ax.bar(x-w/2,rho_eq,w,label='等权均值',color='#1f77b4')
bars2=ax.bar(x+w/2,rho_w,w,label='素质加权',color='#2ca02c')
for b1,b2 in zip(bars1,bars2):
    ax.text(b1.get_x()+b1.get_width()/2,b1.get_height()+0.01,f'{b1.get_height():.3f}',ha='center',fontsize=9)
    ax.text(b2.get_x()+b2.get_width()/2,b2.get_height()+0.01,f'{b2.get_height():.3f}',ha='center',fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(TOPIC_LABELS)
ax.set_ylabel('Spearman ρ (vs 奖项)'); ax.set_title('等权 vs 素质加权: 网评与奖项相关性'); ax.legend(loc='upper left')
save_figure(fig,'q5-weighted-comparison',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题5：等权vs素质加权Spearman ρ对比',review=True)

# --- Q5-3: 降权建议 ---
demotions={'A':0,'B':6,'C':0,'D':7,'E':0}
total={t:len(q3df[q3df['题目']==t]) for t in TOPICS}
fig,ax=plt.subplots(figsize=(7,4))
x=np.arange(5)
bars1=ax.bar(x,[total[t] for t in TOPICS],label='正常权重',color='#1f77b4',alpha=0.5)
bars2=ax.bar(x,[demotions[t] for t in TOPICS],label='建议降权',color='#d62728',alpha=0.8,width=0.5)
for i,t in enumerate(TOPICS):
    if demotions[t]>0:
        ax.text(i,demotions[t]+1,f'{demotions[t]}位',ha='center',fontsize=10,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(TOPIC_LABELS)
ax.set_ylabel('评委数'); ax.set_title('建议降权评委数量')
ax.legend()
save_figure(fig,'q5-demotion-suggestion',fig_dir=FIG_DIR,chart_dir=CHART_DIR,
            context='子问题5：建议降权评委按题目汇总',review=True)

print('\n===== 全部图表生成完毕 =====')
print(f'FIG_DIR: {FIG_DIR}')
print(f'CHART_DIR: {CHART_DIR}')
