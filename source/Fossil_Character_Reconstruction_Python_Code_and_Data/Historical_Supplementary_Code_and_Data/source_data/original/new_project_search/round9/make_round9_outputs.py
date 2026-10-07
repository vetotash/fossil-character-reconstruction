from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R=Path(__file__).resolve().parent
O=R/'deliverables';O.mkdir(exist_ok=True)
a=pd.read_csv(R/'taxon_ablation/summary.csv')
f=pd.read_csv(R/'character_robustness/full_summary.csv')
assert len(a)==138 and len(f)==42
schemes=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
labels=['Frequency','kNN','Logistic','Extra Trees','Mk marginal','Mk joint']
colors=['#263238','#0072B2','#D55E00','#009E73','#9B59B6']
fig,axs=plt.subplots(1,3,figsize=(14.7,9.3))
plotrows=[]
for ax,history,ref,title,names in zip(axs,['tetrapods','dinosaurs','dinosaurs'],['O1','baron','langer'],['A  Tetrapods, O1','B  Dinosaurs, Baron reference','C  Dinosaurs, Langer reference'],[['Lethiscus','Ossinodus','Ventastega'],['Liliensternus','Syntarsus','Eodromaeus','original_question_mark_only'],['Liliensternus','Syntarsus','Eodromaeus','original_question_mark_only']]):
    scenarios=['Full cohort']+names
    for mi,method in enumerate(schemes):
        for si,scenario in enumerate(scenarios):
            if si==0:
                z=f[(f.lineage==history)&(f.reference==ref)&(f.scheme==method)].iloc[0]
                obs=z.observed_excess;mean=z.sim_mean;lo=z.sim_q025;hi=z.sim_q975;n=int(z.active_cells)
            else:
                z=a[(a.history==history)&(a.reference==ref)&(a.scheme==method)&(a.focal_taxon==scenario)].iloc[0]
                obs=z.reference_excess_steps;mean=z.sim_mean_excess_steps;lo=z.sim_q025;hi=z.sim_q975;n=int(z.remaining_target_cells)
            y=mi*1.35+(si-(len(scenarios)-1)/2)*.19
            ax.plot([lo-obs,hi-obs],[y,y],lw=1.5,color=colors[si])
            ax.plot(mean-obs,y,'o',ms=4,color=colors[si])
            plotrows.append(dict(history=history,reference=ref,scheme=method,scenario=scenario,targets=n,reference_excess=float(obs),mean=float(mean),q025=float(lo),q975=float(hi)))
    ax.axvline(0,color='#555555',lw=.9,ls='--')
    ax.set_yticks(np.arange(6)*1.35,labels)
    ax.invert_yaxis();ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=14)
    ax.grid(axis='x',color='#dddddd',lw=.6);ax.set_axisbelow(True)
    ax.set_xlabel('Simulated excess steps minus\nlater published reference',fontsize=10)
    for spine in ['right','top']:ax.spines[spine].set_visible(False)
    leglabels=['Full cohort']+[('Original ? only' if x=='original_question_mark_only' else f'{x} targets unfilled') for x in names]
    handles=[Line2D([0],[0],color=colors[j],marker='o',ms=4,lw=1.5,label=t) for j,t in enumerate(leglabels)]
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,-.12),fontsize=8.5,frameon=False,handlelength=2)
    ax.tick_params(labelsize=9)
fig.subplots_adjust(left=.065,right=.99,top=.9,bottom=.29,wspace=.42)
fig.suptitle('Historical scoring diagnostics depend on the target set',fontsize=15,y=.97)
fig.text(.065,.025,'Points: means. Lines: central 95% conditional simulation ranges from 1,000 frozen draws. Zero: each scenario\'s own later-score endpoint.\nTargets are left unresolved; taxa and topologies are retained. O1 is illustrated; all five tetrapod references are reported in the data.',fontsize=9,color='#444444')
fig.savefig(O/'historical_scoring_sensitivity.png',dpi=200)
fig.savefig(O/'historical_scoring_sensitivity.pdf')
plt.close(fig)
pd.DataFrame(plotrows).to_csv(O/'figure_sensitivity_data.csv',index=False)
strict=a[a.focal_taxon=='original_question_mark_only'].copy()
strict.to_csv(O/'original_question_mark_sensitivity.csv',index=False)
changes=a[a.interval_status_changed].copy();changes.to_csv(O/'all_interval_status_changes.csv',index=False)
caption='''Figure. Historical scoring diagnostics depend on the target set. Points and horizontal lines represent the mean and central 95% empirical range of 1,000 propagated completions, after subtracting the corresponding later published reference increment. A: tetrapod reference O1. B and C: the two 2023 dinosaur reference topologies. Leaving targets unfilled retains every taxon, original observed score, frozen probability and topology; it is not taxon pruning or model refitting. The original-question-mark sensitivity leaves 25 originally dashed dinosaur cells unresolved while retaining 1,308 cells printed as question marks. Each scenario has its own later-score reference, so the zero lines do not imply identical absolute reference costs. Simulation ranges are conditional diagnostics, not confidence intervals or estimates of repeated coverage. All five tetrapod references and both dinosaur references are included in the full numerical tables; O1 is the illustrated tetrapod example.\n'''
(O/'figure_caption.txt').write_text(caption)
report='''化石形态历史预测项目 第九轮核验报告
2026年10月3日

本轮判断
已经形成可以继续完善的实证稿件。新增核验增强了可追溯性，也否定了“某方法普遍更可靠”的简单结论。现阶段可支持的主题是：在两段真实化石矩阵修订历史中，单一预测值与概率传播会给出不同的下游性状冲突评价；这种评价还取决于具体类群、字符和原始符号处理。

英文题目
Historical Tests of Probabilistic Character Reconstruction in Fossil Vertebrates

数据和分母
四足类：早期矩阵102类群、339字符。既有接受映射内有3005个旧缺失格；旧训练量门槛后有2835格，其中350格后来成为单态评分。95类群的下游面板保留349格、87字符。350个目标的原初选择代码未恢复，门槛规则能重建其集合，但不能声称原始整理过程已端到端复现。
恐龙：75类群、457字符的旧预测矩阵，74共享类群用于下游分析。存档中训练准入的16929个缺失格产生1333个后来单态目标。核查原论文PDF发现，这16929格由16729个“?”和200个“−”组成；1333个目标中1308个原为“?”，25个原为“−”。保留两个符号的区别；没有证据将25格逐一认证为生物学不适用。
原始“?”敏感性只在下游留空25格，保持全部旧训练、类群、树和概率不变；它不重新拟合，也不声称恢复了以另一套原始符号建模的全过程。

本轮实际完成
1. 全部7014项逐一去除字符检验；168项去除最大绝对残差字符的压力测试；42项均值方向脆弱性分析。
2. 6个目标最多类群各自留空的原生PAUP重算，以及1项恐龙原始符号敏感性。保留全部六种传播方案、每项1000次完成，共138项汇总、138000个逐次结果。没有把类群贡献假定为可加后直接减分。
3. 四足类349格的来源登记和87字符解剖分区；本轮直接重读58格，其他291格继承此前审计且已标记。恐龙全部1333格的原始符号对照、10个重点字符定义及18个优先格的原文依据。
4. 全部16270条历史预测的概率、类别、最大概率标签与训练量核查；原恐龙1221个控制选择身份及顺序完全复现。本轮没有重新拟合模型。

关键数值
只含原始问号的1308格恐龙分析，后来评分的新增超额步数从221/191变为219/189。六方案乘两参考树共12项区间相对位置全部不变。频率和Extra Trees仍偏高；kNN与logistic的区间仍包含后来评分；Mk两种传播在Baron参考上包含后来评分，在Langer参考上偏高。这不是校准覆盖率证明。

类群留空导致126项比较中的15项区间判断变化，不能把全队列的区间包含关系写成稳定规律。四足类Ventastega的24格留空后，O1后来参考由37变36：Extra Trees均值44.388、区间34–55，参考进入区间；联合Mk均值25.452、区间17–34，参考反而落到区间之上。O2也有对应变化。恐龙Liliensternus的112格留空后，Baron参考为212，而kNN和logistic区间分别169–206、169.975–205，原先包含参考的结论不再成立。
频率基线在全部字符和类群压力测试下仍高于后来参考。Extra Trees的偏高方向在各类群检验中保留，但四足类Ventastega例不再排除参考；“方向保留”和“区间排除”必须分开。

解剖解释
Lethiscus的犁骨接触评分在2019附录中带有明确不确定性；Schoenfelderpeton的后冠骨齿涉及对旧图示和大小阈值的重新解释；Ossinodus的肱骨突起依据后来解剖描述。这些使研究有具体古生物学对象，但后来发表不等于后来新发现。
Lethiscus相关CT数据始于1998年，后来重新处理；Ventastega相关新描述材料至迟于2001年已采集。不能写成2007年后才发现的新标本。
恐龙的肩胛骨、髂骨、耻骨和颈椎案例能追溯到修订者列明的馆藏材料。18个优先格有16个明确修改条目，2个只由矩阵确认。另发现Liliensternus字符321的清单和矩阵均为1，而讨论描述对应2；保留原发布分数并登记矛盾，未擅自改分。

方法与复现边界
比较三种机器学习方法：kNN、logistic、Extra Trees；频率和NJ加Mk为基线。联合Mk是同一已拟合模型的另一种传播方式。新增四叶树例子仅说明：同样的单格概率，不足以唯一决定性状冲突的期望；它不是新定理或化石实证。
当前能从冻结矩阵和目标/控制清单重新拟合两套面板，并提供准确命令；本轮执行的是输出审计与下游复算，不能把提供命令写成已完成全量重新拟合。四足类初始控制选择代码尚缺；此前logistic未保存逐次收敛记录。这些限制已进入英文稿。
PAUP计算维持原字符代价和多态/不确定性设置，每次扣除其本身的最小步数变化。目标是旧背景下部分补全的新增超额步数，不是真实趋同次数，也不是整套矩阵修订或新树搜索的效果。

论文定位
主体仍是古脊椎动物形态证据，交叉内容是分类预测、概率传播和来源审计。主要贡献是贯通“旧矩阵缺失、后来评分、同一背景下的系统发育兼容性代价”，并如实展示失败与敏感性。已有研究已经做过缺失替换、算法比较、模型适合度及字符信息量，不能把这些本身作为首创。
两段历史及相关参考树不足以宣称跨整个显生宙的一般规律，目前也没有充分证据认证稿件达到指定一区刊的接收标准。可以写成实证稿，不等于保证一区发表。具体价值应由贡献、独立历史数量和原始证据质量评估，不用增加算法数量代替这些问题。

交付说明
英文稿已整合新的来源、分母、方法和稳健性结果，保留此前完整42项主结果和约束树未反转的负结果。数值表、逐格记录、冻结输入与代码随压缩包提供；第三方完整论文和受许可限制的软件不随包重复分发，来源URL和哈希保留。
'''
(O/'fossil_morphology_round9_report.txt').write_text(report,encoding='utf-8')
print(json.dumps({'summary_rows':len(a),'status_changes':len(changes),'strict_rows':len(strict),'strict_changes':int(strict.interval_status_changed.sum()),'figure_rows':len(plotrows),'outputs':[p.name for p in sorted(O.iterdir())]},ensure_ascii=False,indent=2))
