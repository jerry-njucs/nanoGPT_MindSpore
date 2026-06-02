三步走：one-eight-fifteen

使用提供的词表

预处理命令行和输出：
(nanogpt) PS D:\OneDrive - 南京大学\2025-2026 秋 大三上学期\智能计算系统\exp4\nanoGPT_MindSpore> python src/preprocess.py --jinyong_dir ./data/jinyong_one --tokenizer_path ./dataset/chinese_char_tokenizer.json --output_file ./dataset/jinyong_one --file_partition 2
加载 tokenizer: ./dataset/chinese_char_tokenizer.json
Tokenizer 已从 ./dataset/chinese_char_tokenizer.json 加载
  - 词表大小: 5424

Tokenizer 信息:
  vocab_size: 8000
  actual_vocab_size: 5424
  special_tokens: 4
  regular_chars: 5420
  pad_id: 0
  unk_id: 1
  bos_id: 2
  eos_id: 3

读取数据: ./data/jinyong_one
总字符数: 965,744
编码文本...
总 token 数: 965,744
压缩比: 1.000

生成序列 (SEQ_LEN=256)...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3772/3772 [00:03<00:00, 1056.72it/s]
Transformed 3772 records.
Transform finished, output files refer: ./dataset/jinyong_one0

处理图片已保存


local test作为云训练前本地的smoke test执行


smoke test成功，调整参数之后可以开始云训练



一本书：
(MindSpore) [ma-user nanogpt]$python train.py --data_path dataset/new-jinyong-one --epoch 2 --batch_size 12 --learning_rate 6e-4 --min_lr 6e-5 --warmup_iters 500 --lr_decay_iters 10000 --sink_size 100
/home/ma-user/anaconda3/envs/MindSpore/lib/python3.10/site-packages/numpy/core/getlimits.py:499: UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
  setattr(self, word, getattr(machar, word).flat[0])
/home/ma-user/anaconda3/envs/MindSpore/lib/python3.10/site-packages/numpy/core/getlimits.py:89: UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
  return self._float_to_str(self.smallest_subnormal)
/home/ma-user/anaconda3/envs/MindSpore/lib/python3.10/site-packages/numpy/core/getlimits.py:499: UserWarning: The value of the smallest subnormal for <class 'numpy.float32'> type is zero.
  setattr(self, word, getattr(machar, word).flat[0])
/home/ma-user/anaconda3/envs/MindSpore/lib/python3.10/site-packages/numpy/core/getlimits.py:89: UserWarning: The value of the smallest subnormal for <class 'numpy.float32'> type is zero.
  return self._float_to_str(self.smallest_subnormal)
[WARNING] CORE(92920,ffffb6d35010,python):2025-12-15-11:07:34.736.097 [mindspore/core/utils/ms_context.cc:530] GetJitLevel] Set jit level to O2 for rank table startup method.
['/home/ma-user/work/nanogpt/dataset/new-jinyong-one/mindrecord']
crated dataset
step_per_epoch: 314
num decayed parameter tensors: 51, with 94,076,928 parameters
num non-decayed parameter tensors: 25, with 19,200 parameters
start training
.....epoch: 1 step: 100, loss is 5.411800384521484
Train epoch time: 66471.205 ms, per step time: 664.712 ms
epoch: 2 step: 100, loss is 4.958648681640625
Train epoch time: 15999.017 ms, per step time: 159.990 ms
epoch: 3 step: 100, loss is 4.489719867706299
Train epoch time: 16909.526 ms, per step time: 169.095 ms
epoch: 4 step: 100, loss is 4.691559314727783
Train epoch time: 17526.519 ms, per step time: 175.265 ms
epoch: 5 step: 100, loss is 4.414837837219238
Train epoch time: 15873.941 ms, per step time: 158.739 ms
epoch: 6 step: 100, loss is 4.4091644287109375
Train epoch time: 15925.233 ms, per step time: 159.252 ms


生成效果：
那人一掌击出
.
==================================================
输入: 那人一掌击出
==================================================
生成: 那人一掌击出来，再加上了这么？”郭靖道：“我不是我还是我师哥。”黄蓉笑道：“你们自然，你不见？”郭靖道：“你这小姑的手里？”



　　黄蓉笑道：“傻姑，我是一掌法子，你侄，还不过，那还要不要是要我总不得出来。”黄蓉在此事儿，但要你在一掌里的穴道长的手。”黄蓉摇头而出来，转身子微一声，心想：“你爹爹爹爹，那么说，一样。”黄蓉道：“我若是不是非一时，他们是不得。他也不见黄药师的“这里还是我到了一个么，这几句话还
==================================================

输入一个开头：只见一位白衣少女

==================================================
输入: 只见一位白衣少女
==================================================
生成: 只见一位白衣少女，忙声中大笑，竟似乎有的手里。杨康心知他无，却不到他笑，笑道：“啊哟，不是甚么？”


　　穆念慈道：“你们这是不是不知，这里来。”黄蓉道：“你不到我也去就能将，今日在旁一个儿。”黄蓉大喜，一声喝道：“我不用，你来打我的事好好。”



　　欧阳锋在地下招“我的，不见欧阳锋本事，不觉他身上的双腕上，这里有脚一响，心中的招亲”的掌相劝，身前一句。欧阳锋见他笑道：“你来也打个一个女。”欧阳锋大惊，右手
==================================================



一本 epoch=10
start training
.epoch: 1 step: 100, loss is 5.411800384521484
Train epoch time: 27350.485 ms, per step time: 273.505 ms
epoch: 2 step: 100, loss is 4.958648681640625
Train epoch time: 18193.359 ms, per step time: 181.934 ms
epoch: 3 step: 100, loss is 4.489701271057129
Train epoch time: 15488.158 ms, per step time: 154.882 ms
epoch: 4 step: 100, loss is 4.692634582519531
Train epoch time: 17926.263 ms, per step time: 179.263 ms
epoch: 5 step: 100, loss is 4.412991046905518
Train epoch time: 15979.635 ms, per step time: 159.796 ms
epoch: 6 step: 100, loss is 4.415408611297607
Train epoch time: 15935.713 ms, per step time: 159.357 ms
epoch: 7 step: 100, loss is 3.8255245685577393
Train epoch time: 18085.027 ms, per step time: 180.850 ms
epoch: 8 step: 100, loss is 4.047706604003906
Train epoch time: 15773.567 ms, per step time: 157.736 ms
epoch: 9 step: 100, loss is 3.824601650238037
Train epoch time: 15835.922 ms, per step time: 158.359 ms
epoch: 10 step: 100, loss is 3.6911988258361816
Train epoch time: 17627.353 ms, per step time: 176.274 ms
epoch: 11 step: 100, loss is 3.5478310585021973
Train epoch time: 14897.425 ms, per step time: 148.974 ms
epoch: 12 step: 100, loss is 3.521679639816284
Train epoch time: 15125.139 ms, per step time: 151.251 ms
epoch: 13 step: 100, loss is 3.2095415592193604
Train epoch time: 21107.451 ms, per step time: 211.075 ms
epoch: 14 step: 100, loss is 3.0804412364959717
Train epoch time: 19267.613 ms, per step time: 192.676 ms
epoch: 15 step: 100, loss is 3.2142062187194824
Train epoch time: 19384.094 ms, per step time: 193.841 ms
epoch: 16 step: 100, loss is 2.822800397872925
Train epoch time: 23589.241 ms, per step time: 235.892 ms
epoch: 17 step: 100, loss is 2.8985817432403564
Train epoch time: 21216.087 ms, per step time: 212.161 ms
epoch: 18 step: 100, loss is 2.803147554397583
Train epoch time: 20842.004 ms, per step time: 208.420 ms
epoch: 19 step: 100, loss is 2.279494047164917
Train epoch time: 22773.774 ms, per step time: 227.738 ms
epoch: 20 step: 100, loss is 2.351968288421631
Train epoch time: 20756.069 ms, per step time: 207.561 ms
epoch: 21 step: 100, loss is 2.544004440307617
Train epoch time: 20426.773 ms, per step time: 204.268 ms
epoch: 22 step: 100, loss is 1.6533656120300293
Train epoch time: 19148.188 ms, per step time: 191.482 ms
epoch: 23 step: 100, loss is 1.7959855794906616
Train epoch time: 18211.774 ms, per step time: 182.118 ms
epoch: 24 step: 100, loss is 1.8739947080612183
Train epoch time: 19583.042 ms, per step time: 195.830 ms
epoch: 25 step: 100, loss is 2.1019160747528076
Train epoch time: 20136.529 ms, per step time: 201.365 ms
epoch: 26 step: 100, loss is 1.1473722457885742
Train epoch time: 22997.057 ms, per step time: 229.971 ms
epoch: 27 step: 100, loss is 1.235395908355713
Train epoch time: 20297.685 ms, per step time: 202.977 ms
epoch: 28 step: 100, loss is 1.406522274017334
Train epoch time: 19851.758 ms, per step time: 198.518 ms
epoch: 29 step: 100, loss is 0.6923470497131348
Train epoch time: 24570.286 ms, per step time: 245.703 ms
epoch: 30 step: 100, loss is 0.8579543828964233
Train epoch time: 17154.632 ms, per step time: 171.546 ms
epoch: 31 step: 100, loss is 0.887926459312439
Train epoch time: 19222.860 ms, per step time: 192.229 ms


输入一个开头：那人一掌击出
.
==================================================
输入: 那人一掌击出
==================================================
生成: 那人一掌击出面又是这等人抹苦。



　　黄蓉回了头，问道：“靖哥哥，咱们在这里胡说八道算帐，你别听我说的？”郭靖甚是得意思，哪想得够疑心，甚是得意，忽然痴呆了，又向黄蓉道：“我到王师父陪教你三位师父说，该当如何？”黄蓉叹道：“说来话三师父，可是我不是！”从怀中瞧他一看，见是大大的羞头，心刻总是盼助对父赐言：“我们的弟子都识得来，岂难称你们另有甚么事？”



　　正自怔了下头，忽听崖边一个破钹的声音在一排沙
==================================================

输入一个开头：只见一位白衣少女

==================================================
输入: 只见一位白衣少女
==================================================
生成: 只见一位白衣少女，脸上惊怒，心道：“小姑娘当真叫作浪魂西北方面，见怪着得人！”他双手一拱，但随即已从吟处下楼板至背脊，这一下“穿阳子”字诀连遭向她不见，确是大出惊诧，但见她自己头皮和窝阔台相距不远，当即飞身而起，双臂被点中了一枝白刃，正要跌下公子忽施袭击他背心，惊叫：“大师父，你师父快来，快乘坐我！”



　　此言一出，一骑，四师父也已赶上，都行到了北郊外，他正危急之际，忽见巨路尽横抱着一个老儿手臂，一瞬之间，
==================================================



8本书 epoch=4

start training
.epoch: 1 step: 100, loss is 5.780733108520508
Train epoch time: 26484.429 ms, per step time: 264.844 ms
epoch: 2 step: 100, loss is 4.862179756164551
Train epoch time: 16402.588 ms, per step time: 164.026 ms
epoch: 3 step: 100, loss is 4.763622283935547
Train epoch time: 15993.517 ms, per step time: 159.935 ms
epoch: 4 step: 100, loss is 4.693564414978027
Train epoch time: 16126.966 ms, per step time: 161.270 ms
epoch: 5 step: 100, loss is 4.6058878898620605
Train epoch time: 17247.361 ms, per step time: 172.474 ms
epoch: 6 step: 100, loss is 4.340768814086914
Train epoch time: 16318.267 ms, per step time: 163.183 ms
epoch: 7 step: 100, loss is 4.48962926864624
Train epoch time: 18412.732 ms, per step time: 184.127 ms
epoch: 8 step: 100, loss is 4.515542507171631
Train epoch time: 19685.298 ms, per step time: 196.853 ms
epoch: 9 step: 100, loss is 4.118881702423096
Train epoch time: 21330.819 ms, per step time: 213.308 ms
epoch: 10 step: 100, loss is 4.084808349609375
Train epoch time: 22172.819 ms, per step time: 221.728 ms
epoch: 11 step: 100, loss is 4.011285305023193
Train epoch time: 22099.729 ms, per step time: 220.997 ms
epoch: 12 step: 100, loss is 3.632322311401367
Train epoch time: 21108.790 ms, per step time: 211.088 ms
epoch: 13 step: 100, loss is 3.7118489742279053
Train epoch time: 17422.595 ms, per step time: 174.226 ms
epoch: 14 step: 100, loss is 3.780057430267334
Train epoch time: 16742.783 ms, per step time: 167.428 ms
epoch: 15 step: 100, loss is 3.7127983570098877
Train epoch time: 16600.459 ms, per step time: 166.005 ms
epoch: 16 step: 100, loss is 3.6737594604492188
Train epoch time: 17147.157 ms, per step time: 171.472 ms
epoch: 17 step: 100, loss is 3.5669660568237305
Train epoch time: 16861.128 ms, per step time: 168.611 ms
epoch: 18 step: 100, loss is 3.724395990371704
Train epoch time: 16867.840 ms, per step time: 168.678 ms
epoch: 19 step: 100, loss is 3.4172303676605225
Train epoch time: 16972.376 ms, per step time: 169.724 ms
epoch: 20 step: 100, loss is 3.3647921085357666
Train epoch time: 16795.317 ms, per step time: 167.953 ms
epoch: 21 step: 100, loss is 3.5663814544677734
Train epoch time: 16853.667 ms, per step time: 168.537 ms
epoch: 22 step: 100, loss is 3.580172538757324
Train epoch time: 17383.380 ms, per step time: 173.834 ms
epoch: 23 step: 100, loss is 3.300612211227417
Train epoch time: 19903.903 ms, per step time: 199.039 ms
epoch: 24 step: 100, loss is 3.376664638519287
Train epoch time: 16890.319 ms, per step time: 168.903 ms
epoch: 25 step: 100, loss is 3.441211700439453
Train epoch time: 17001.087 ms, per step time: 170.011 ms
epoch: 26 step: 100, loss is 3.13926362991333
Train epoch time: 17248.211 ms, per step time: 172.482 ms
epoch: 27 step: 100, loss is 3.138540029525757
Train epoch time: 16861.415 ms, per step time: 168.614 ms
epoch: 28 step: 100, loss is 3.1462597846984863
Train epoch time: 20161.284 ms, per step time: 201.613 ms
epoch: 29 step: 100, loss is 3.178682327270508
Train epoch time: 16760.159 ms, per step time: 167.602 ms
epoch: 30 step: 100, loss is 3.370605707168579
Train epoch time: 16902.872 ms, per step time: 169.029 ms
epoch: 31 step: 100, loss is 3.3177504539489746
Train epoch time: 16363.904 ms, per step time: 163.639 ms
epoch: 32 step: 100, loss is 3.1802964210510254
Train epoch time: 16546.898 ms, per step time: 165.469 ms
epoch: 33 step: 100, loss is 3.101766347885132
Train epoch time: 16824.370 ms, per step time: 168.244 ms
epoch: 34 step: 100, loss is 3.038295030593872
Train epoch time: 20265.920 ms, per step time: 202.659 ms
epoch: 35 step: 100, loss is 3.431241035461426
Train epoch time: 20092.801 ms, per step time: 200.928 ms
epoch: 36 step: 100, loss is 3.2374846935272217
Train epoch time: 16142.808 ms, per step time: 161.428 ms
epoch: 37 step: 100, loss is 3.1154115200042725
Train epoch time: 19194.936 ms, per step time: 191.949 ms
epoch: 38 step: 100, loss is 3.0226943492889404
Train epoch time: 19229.235 ms, per step time: 192.292 ms
epoch: 39 step: 100, loss is 3.1794846057891846
Train epoch time: 17263.864 ms, per step time: 172.639 ms
epoch: 40 step: 100, loss is 3.2505388259887695
Train epoch time: 20289.203 ms, per step time: 202.892 ms
epoch: 41 step: 100, loss is 3.094646692276001
Train epoch time: 20197.026 ms, per step time: 201.970 ms
epoch: 42 step: 100, loss is 3.19209361076355
Train epoch time: 18490.732 ms, per step time: 184.907 ms
epoch: 43 step: 100, loss is 3.0890283584594727
Train epoch time: 20138.644 ms, per step time: 201.386 ms
epoch: 44 step: 100, loss is 3.16015887260437
Train epoch time: 16278.165 ms, per step time: 162.782 ms
epoch: 45 step: 100, loss is 2.9058475494384766
Train epoch time: 18161.028 ms, per step time: 181.610 ms
epoch: 46 step: 100, loss is 2.8380396366119385
Train epoch time: 21639.316 ms, per step time: 216.393 ms
epoch: 47 step: 100, loss is 2.643136739730835
Train epoch time: 16354.279 ms, per step time: 163.543 ms
epoch: 48 step: 100, loss is 2.892364978790283
Train epoch time: 16427.644 ms, per step time: 164.276 ms
epoch: 49 step: 100, loss is 2.7480390071868896
Train epoch time: 17701.078 ms, per step time: 177.011 ms
epoch: 50 step: 100, loss is 2.8681836128234863
Train epoch time: 18609.231 ms, per step time: 186.092 ms
epoch: 51 step: 100, loss is 2.919322967529297
Train epoch time: 18189.349 ms, per step time: 181.893 ms
epoch: 52 step: 100, loss is 2.8358263969421387
Train epoch time: 15879.083 ms, per step time: 158.791 ms
epoch: 53 step: 100, loss is 2.8053863048553467
Train epoch time: 18754.416 ms, per step time: 187.544 ms
epoch: 54 step: 100, loss is 2.777984857559204
Train epoch time: 18771.027 ms, per step time: 187.710 ms
epoch: 55 step: 100, loss is 2.758676052093506
Train epoch time: 18842.506 ms, per step time: 188.425 ms
epoch: 56 step: 100, loss is 2.9190735816955566
Train epoch time: 18387.914 ms, per step time: 183.879 ms
epoch: 57 step: 100, loss is 2.956817388534546
Train epoch time: 16927.704 ms, per step time: 169.277 ms
epoch: 58 step: 100, loss is 2.754194974899292
Train epoch time: 17048.361 ms, per step time: 170.484 ms
epoch: 59 step: 100, loss is 2.5702109336853027
Train epoch time: 17065.780 ms, per step time: 170.658 ms
epoch: 60 step: 100, loss is 2.6965272426605225
Train epoch time: 16554.162 ms, per step time: 165.542 ms
epoch: 61 step: 100, loss is 2.9259848594665527
Train epoch time: 18550.854 ms, per step time: 185.509 ms
epoch: 62 step: 100, loss is 2.7010154724121094
Train epoch time: 16356.987 ms, per step time: 163.570 ms
epoch: 63 step: 100, loss is 2.7765488624572754
Train epoch time: 16480.712 ms, per step time: 164.807 ms
epoch: 64 step: 100, loss is 2.737610340118408
Train epoch time: 16114.037 ms, per step time: 161.140 ms
epoch: 65 step: 100, loss is 2.6226634979248047
Train epoch time: 16919.370 ms, per step time: 169.194 ms
epoch: 66 step: 100, loss is 2.6401450634002686
Train epoch time: 15814.311 ms, per step time: 158.143 ms
epoch: 67 step: 100, loss is 2.4574062824249268
Train epoch time: 17830.698 ms, per step time: 178.307 ms
epoch: 68 step: 100, loss is 2.6448769569396973
Train epoch time: 16450.508 ms, per step time: 164.505 ms
epoch: 69 step: 100, loss is 2.4384729862213135
Train epoch time: 17904.963 ms, per step time: 179.050 ms
epoch: 70 step: 100, loss is 2.4311652183532715
Train epoch time: 16148.683 ms, per step time: 161.487 ms
epoch: 71 step: 100, loss is 2.388207197189331
Train epoch time: 16178.205 ms, per step time: 161.782 ms
epoch: 72 step: 100, loss is 2.3255107402801514
Train epoch time: 15996.097 ms, per step time: 159.961 ms
epoch: 73 step: 100, loss is 2.272596836090088
Train epoch time: 16333.449 ms, per step time: 163.334 ms
epoch: 74 step: 100, loss is 2.425628185272217
Train epoch time: 15286.328 ms, per step time: 152.863 ms
epoch: 75 step: 100, loss is 2.3405299186706543
Train epoch time: 15687.617 ms, per step time: 156.876 ms
epoch: 76 step: 100, loss is 2.5507001876831055
Train epoch time: 17583.582 ms, per step time: 175.836 ms
epoch: 77 step: 100, loss is 2.213949680328369
Train epoch time: 16691.143 ms, per step time: 166.911 ms
epoch: 78 step: 100, loss is 2.3774893283843994
Train epoch time: 16401.345 ms, per step time: 164.013 ms
epoch: 79 step: 100, loss is 2.4541385173797607
Train epoch time: 15941.636 ms, per step time: 159.416 ms
epoch: 80 step: 100, loss is 2.400021553039551
Train epoch time: 17710.273 ms, per step time: 177.103 ms
epoch: 81 step: 100, loss is 2.562502861022949
Train epoch time: 18078.120 ms, per step time: 180.781 ms
epoch: 82 step: 100, loss is 2.392462968826294
Train epoch time: 17430.814 ms, per step time: 174.308 ms
epoch: 83 step: 100, loss is 2.5349631309509277
Train epoch time: 15800.641 ms, per step time: 158.006 ms
epoch: 84 step: 100, loss is 2.08288311958313
Train epoch time: 17283.234 ms, per step time: 172.832 ms
epoch: 85 step: 100, loss is 2.34012508392334
Train epoch time: 17667.158 ms, per step time: 176.672 ms
epoch: 86 step: 100, loss is 2.2174856662750244
Train epoch time: 15838.198 ms, per step time: 158.382 ms
epoch: 87 step: 100, loss is 2.4355409145355225
Train epoch time: 17743.706 ms, per step time: 177.437 ms
epoch: 88 step: 100, loss is 2.248443841934204
Train epoch time: 17262.570 ms, per step time: 172.626 ms
epoch: 89 step: 100, loss is 2.3664541244506836
Train epoch time: 16134.046 ms, per step time: 161.340 ms
epoch: 90 step: 100, loss is 2.3170130252838135
Train epoch time: 16253.912 ms, per step time: 162.539 ms



==================================================
输入: 那人一掌击出
==================================================
生成: 那人一掌击出，噗的一声，一件细衣护身的小蛇头顶已中下木板。这条蛇身材长大，但带着一条条小腹，便与一条绿色长袍的红光闪闪的短蛇缠住了一条长索。



　　黑玫瑰被那人左手抓住，被扯去伤口，登时晕倒，眼见天色将黑，这黑衣童子已然毙命，心想：“这人的武功果然了得。”回过身来，只见段誉双手握着绿竹杖，脸颊红肿，容色憔悴，双目深陷，心中虽然充满了怜爱情意，总是无可奈何，但想到这种人之中，实是难以形容。



　　那男童
==================================================

输入一个开头：只见一位白衣少女

==================================================
输入: 只见一位白衣少女
==================================================
生成: 只见一位白衣少女年纪，一双黄眼发胡，满脸堆欢，容色间颇有异样。



　　众人见她对这少女十分关怀，心下甚感歉仄，但想：“她为什么要这么好看？”



　　那少女向他凝视半晌，问道：“你是什么人？是我甚么人？”那少女道：“是我。”那少女道：“你别问我了。你问我为甚么不答允？你在我肩上一打。”那少女道：“你是受了谁的的指使？”



　　那少女道：“我问你：我说的是人是鬼。”那少妇脸上一红，道：“你说的是‘你’？”
==================================================


temperature 0.75 top_k 40 better

==================================================
输入: 那人一掌击出
==================================================
生成: 那人一掌击出，一股内力向外猛击而至，这是为了对方身上中拳，如何抵御，只须运气，立失不得，当即运功抵御，岂知那人竟是毫无容情。他一掌将那人胸口推得隐隐生疼，手掌忽然之间，不由自主的一掌拍出，又已击中他的左肩。



　　这一下手掌力道甚深，但他体内内力已然大致，这一掌出尽，内力又是一股大力，若非真气运转，非有今日之意，非死即伤。他本门弟子虽然武功精湛，究竟是不弱于人、无量剑的高手，自己若再相救，决无大碍。只因这人内力一发，内力竟然全然不减，心中一惊，只觉全身酸软，说不出的舒畅。



　　那人身形一晃，已摔在两名家，脸色无可再无比虚丑陋得多，这两只是他不及时光。

输入: 只见一位白衣少女
==================================================
生成: 只见一位白衣少女站在他身旁，心下暗暗为难，当下微微点了点头，说道：“你不是我妈妈，我在你身旁，你们大家不可多问。”那少女道：“你们干什么？”



　　那少女又点了点头，点了点头。



　　那少女道：“你快去找妈妈妈妈的。”杨过道：“我妈妈说过的。”那少女道：“你爹爹叫我去给我。”那少女道：“你要我妈妈，是我妈妈妈的妈妈，你叫甚么？”杨过道：“咱们又不是一样。”那少女道：“我妈妈是我爹爹，是我妈妈。”杨过道：“你爹爹叫甚么？”那少女道：“我爹爹叫我妈妈，是你叫的。你这才叫我‘妈妈的’字。”杨过奇道：“我妈是我妈妈妈妈妈妈妈妈妈妈妈妈妈，是你的妈妈妈妈妈妈。”那少女人道：“你叫你叫甚么？”这两字的叫他不是一个
==================================================

输入一个开头：只见一位白衣少女

==================================================
输入: 只见一位白衣少女
==================================================
生成: 只见一位白衣少女，一位美貌少女，一张白脸都白了，瞧瞧他容貌甚是甚是美丽。



　　那少女走到后厅，低声道：“啊哟，原来他们是我爷爷爷。”那少女道：“你是我爷爷，是不是？”韦小宝道：“是啊，你是我师爷？”那少女道：“他是王爷，这位姑娘是我师姊。一位是王爷，一个小孩子。”韦小宝道：“啊，我是你师父，他跟你爹爹不是一起去的，可是我……啊，我……我……却有一个人。”那少女“啊”的一声，哭了起来。



　　韦小宝道：“你跟那小姑娘说。我……我……”心中一荡，说道：“你……你已经死啦，也不知是真是假，你这小女孩，是汉子，我不是汉………………………………………………………………………………………………………………………
==================================================


temperature 0.7 topk 30

输入: 那人一掌击出
==================================================
生成: 那人一掌击出，那人又身子摇晃，身子一偏。他一转身，已然不及了，身子已被他击倒。



　　那人伸足踏在地下，又是一手抓住，但听得两人一齐跌倒，再也不敢再上来救，急忙跃起，只见他满脸喜色，显然是个美貌少女，正是段誉，不知有甚难看，当即跃起，抱起了他，问道：“你怎么了？”段誉道：“我跟你说了。”



　　那人道：“你怎知道？你怎知道？”段誉道：“我不知道。”



　　突然之间，段誉心中一惊，大喜之下，问道：“喂，你在哪里？”段誉道：“我在这里。”



　　那人道：“你……你……你是谁？”段誉道：“我……我……………………………………………………………………………………………………………………………………
==================================================

输入一个开头：只见一位白衣少女

==================================================
输入: 只见一位白衣少女
==================================================
生成: 只见一位白衣少女，一个是中年女子，一个是白须白须的姑娘，一个是白发白须白发，一个是白发白须白骨，另有一个青年男子。只是她眼见她白须白发，满头大汗，眼光中又是欢喜。



　　那老人又道：“我爹爹妈妈叫我去叫你妈妈，你是谁？”



　　那少女问道：“是我妈妈？是谁？”那少女道：“你叫我爹爹？你妈妈叫我妈妈叫甚么名字啊？”那少女道：“你爹爹叫你爹爹，可不是叫我妈妈，是我妈妈。你妈妈叫甚么？”那少女摇头道：“我叫他母亲，你叫甚么名字？”那少女道：“我叫‘小叫化，你怎么叫我叫‘小叫化’？”



　　那少女笑道：“你叫甚么？”



文件
6-100  1book 2epoch
1-31 1book 10epoch
2-90 8book 4epoch
3-59 15book 2epoch