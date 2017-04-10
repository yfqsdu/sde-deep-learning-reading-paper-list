# sde-deep-learning-reading-paper-list

填写须知：
-------

- 深度学习小组paper仓库，每个人即时將所阅读的paper上传到此处分享。
- paper的格式请按照标题1.0下的格式来编辑，并放到相应的标题下，如没有对应的标题可以自建标题。
- 标题下需要添加7个减号（语法规定）。:star:代表重要程度; 加粗请左右歌使用两个星号包裹；
- 标题1.0下已经演示使用方式。
- commit时请填写更新详细描述。

---------------------------------------


# 1 Deep Learning History and Basics

## 1.0 Book

**[1]** Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville. "**Deep learning.**" An MIT Press book. (2015). [[pdf]](http://www.deeplearningbook.org/front_matter.pdf) has no title attribute(**Deep Learning Bible, you can read this book while reading following papers.**) :star::star::star::star::star:

## 1.1 Survey

**[1]** David Silver1, Aja Huang1, Chris J. Maddison1, Arthur Guez1. "**Mastering the game of Go with deep neural networks and tree search.**".Nature 529(7587):484(2016). [[pdf]](http://emotion.psychdept.arizona.edu/Jclub/Silver-et-al.%20Mastering%20the%20game%20of%20Go%20with%20deep%20neural%20networks%20and%20tree%20search+Nature+2016.pdf)(**Introduce a new approachto computer Go that uses ‘value networks’ to evaluate board positions and ‘policy networks’ to select moves.**):star::star:

## 1.2 Deep Neural Network(DBN)

## 1.3 ImageNet Evolution（Deep Learning broke out from here）
 
>After reading above papers, you will have a basic understanding of the Deep Learning history, the basic architectures of Deep Learning model(including CNN, RNN, LSTM) and how deep learning can be applied to natural language processing and predcition recognition issues. The following papers will take you in-depth understanding of the Deep Learning method, Deep Learning in different areas of application and the frontiers. I suggest that you can choose the following papers based on your interests and research direction.

# 2 Deep Learning Method

## 2.1 Model

## 2.2 Optimization

## 2.3 Unsupervised Learning / Deep Generative Model

## 2.4 RNN / Sequence-to-Sequence Model

**[1]** Graves, Alex. "**Generating sequences with recurrent neural networks**." arXiv preprint arXiv:1308.0850 (2013). [[pdf]](http://arxiv.org/pdf/1308.0850) **(LSTM, very nice generating result, show the power of RNN)** :star::star::star::star:

**[2]** Cho, Kyunghyun, et al. "**Learning phrase representations using RNN encoder-decoder for statistical machine translation**." arXiv preprint arXiv:1406.1078 (2014). [[pdf]](http://arxiv.org/pdf/1406.1078) **(First Seq-to-Seq Paper)** :star::star::star::star:

**[3]** Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "**Sequence to sequence learning with neural networks**." Advances in neural information processing systems. 2014. [[pdf]](http://papers.nips.cc/paper/5346-information-based-learning-by-agents-in-unbounded-state-spaces.pdf) **(Outstanding Work)** :star::star::star::star::star:

**[4]** Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. "**Neural Machine Translation by Jointly Learning to Align and Translate**." arXiv preprint arXiv:1409.0473 (2014). [[pdf]](https://arxiv.org/pdf/1409.0473v7.pdf) :star::star::star::star:

**[5]** Vinyals, Oriol, and Quoc Le. "**A neural conversational model**." arXiv preprint arXiv:1506.05869 (2015). [[pdf]](http://arxiv.org/pdf/1506.05869.pdf%20(http://arxiv.org/pdf/1506.05869.pdf)) **(Seq-to-Seq on Chatbot)** :star::star::star:


## 2.5 Neural Turing Machine

## 2.6 Deep Reinforcement Learning

**[1]** Volodymyr Mnih,Koray Kavukcuoglu,David Silver,Alex Graves. "**Playing Atari with Deep Reinforcement Learning.**"	Computer Science:1312.5602(2013). [[pdf]](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Publications_files/dqn.pdf)(**Convolutional neural network.**):star::star::star::star::star:

## 2.7 Deep Transfer Learning / Lifelong Learning / especially for RL

## 2.8 One Shot Deep Learning

# 3 Applications

## 3.1 NLP(Natural Language Processing)

**[1]** Jin Qian, Qi Zhang, Ya Guo, Yaqian Zhou, Xuanjing Huang. "**Generating Abbreviations for Chinese Named Entities Using RecurrentNeural Network with Dynamic Dictionary.**" Conference on Empirical Methods in Natural Language Processing :721-370(2016).[[pdf]](http://anthology.aclweb.org/D/D16/D16-1069.pdf)(**Combines recurrent neural network (RNN) with an architecture determining whether a given sequence of characters can be a word or not.**) :star::star::star:

## 3.2 Fraud Detection

## 3.3 Medication Prediction

**[1]** Edward Choi，Andy Schuetz，Jimeng Sun. "**Doctor AI: Predicting Clinical Events via Recurrent Neural Networks.**" Computer Science arXiv:1511.05942(2015).[[pdf]](http://net.pku.edu.cn/dlib/healthcare/EMR%20event%20sequence/Predicting%20Clinical%20Events%20via%20Recurrent%20Neural%20Networks.pdf)(**Propose a generic predictive model that covers observed medical conditions and medication uses.**) :star::star::star::star:

**[2]** Riccardo Miotto, Li Li, Brian A. Kidd, Joel T. Dudley. "**Deep Patient: An Unsupervised Representation to Predict the Future of Patients from the Electronic Health Records.**" Scientific Reports 6:26094(2016).[[pdf]](http://dudleylab.org/wp-content/uploads/2016/05/Deep-Patient-An-Unsupervised-Representation-to-Predict-the-Future-of-Patients-from-the-Electronic-Health-Records.pdf)(** Provide a machine learning framework for augmenting clinical decision systems.**):star::star::star:

**[3]** Xiaohan Li, Shu Wu, Liang Wang. "**Blood Pressure Prediction via Recurrent Models with Contextual Layer.**" WWW 2017, April 3–7, Perth, Australia(2017).[[pdf]](http://www.shuwu.name/sw/RNN-CL.pdf)(**Propose a novel model named recurrent models with contextual layer.**):star::star::star::star:
 
## 3.4 Location Prediction

**[1]** Qiang Liu, Shu Wu, Liang Wang, Tieniu Tan. "**Predicting the Next Location: A Recurrent Model with Spatial and Temporal
Contexts.**".Thirtieth Aaai Conference on Artificial Intelligence, 194-200(2016).[[pdf]](http://www.shuwu.name/sw/STRNN.pdf)(**Propose ST-RNN model.**):star::star::star::star:

## 3.5 Dynamic Recommender

**[1]** YJ Ko,L Maystre,M Grossglauser. "**Collaborative Recurrent Neural Networks for Dynamic Recommender Systems.**".JMLR: Workshop and Conference Proceedings 60:1–16(2016).[[pdf]](http://jmlr.csail.mit.edu/proceedings/papers/v63/ko101.pdf)(**Propose
a novel, flexible and expressive collaborative sequence model based on recurrent neural networks.**):star::star::star: 

**[2]** Yang Song,Ali Mamdouh Elkahky，XiaoDong He. "**Multi-Rate Deep Learning for Temporal Recommendation.**" International Acm Sigir Conference :909-912(2016).[[pdf]](http://sonyis.me/paperpdf/spr209-song_sigir16.pdf)(**Propose a novel deep
neural network based architecture that models the combination of long-term static and short-term temporal user preferences**) :star::star:

## 3.6 Object Segmentation
## 3.7 Data Imputation
**[1]** Beaulieujones, B. K., and J. H. Moore. "**MISSING DATA IMPUTATION IN THE ELECTRONIC HEALTH RECORD USING DEEPLY LEARNED AUTOENCODERS.**" Pacific Symposium on Biocomputing Pacific Symposium on Biocomputing 22(2016):207.[[pdf]](http://psb.stanford.edu/psb-online/proceedings/psb17/beaulieujones.pdf)(**Proposed a novel method to impute missing data in EHR**):star::star:
