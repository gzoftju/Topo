template_train={}
template_train['source']="<User>: Given a node titled {}, the node connect {}. We need to classify the node into 40 classes: 'Numerical Analysis','Multimedia','Logic in Computer Science','Computers and Society','Cryptography and Security','Distributed, Parallel, and Cluster Computing','Human-Computer Interaction','Computational Engineering, Finance, and Science','Networking and Internet Architecture','Computational Complexity','Artificial Intelligence','Multiagent Systems','General Literature','Neural and Evolutionary Computing','Symbolic Computation','Hardware Architecture','Computer Vision and Pattern Recognition','Graphics','Emerging Technologies','Systems and Control','Computational Geometry','Other Computer Science','Programming Languages','Software Engineering','Machine Learning','Sound','Social and Information Networks','Robotics','Information Theory','Performance','Computation and Language','Information Retrieval','Mathematical Software','Formal Languages and Automata Theory','Data Structures and Algorithms','Operating Systems','Computer Science and Game Theory','Databases','Digital Libraries','Discrete Mathematics'. With the title {}, the node is classified as <Assistant>: {}"
template_train['target'] = "{}"
template_test={}
template_test['source']="<User>: Given a node titled {}, the node connect {}. We need to classify the node into 40 classes: 'Numerical Analysis','Multimedia','Logic in Computer Science','Computers and Society','Cryptography and Security','Distributed, Parallel, and Cluster Computing','Human-Computer Interaction','Computational Engineering, Finance, and Science','Networking and Internet Architecture','Computational Complexity','Artificial Intelligence','Multiagent Systems','General Literature','Neural and Evolutionary Computing','Symbolic Computation','Hardware Architecture','Computer Vision and Pattern Recognition','Graphics','Emerging Technologies','Systems and Control','Computational Geometry','Other Computer Science','Programming Languages','Software Engineering','Machine Learning','Sound','Social and Information Networks','Robotics','Information Theory','Performance','Computation and Language','Information Retrieval','Mathematical Software','Formal Languages and Automata Theory','Data Structures and Algorithms','Operating Systems','Computer Science and Game Theory','Databases','Digital Libraries','Discrete Mathematics'. With the title {}, the node is classified as <Assistant>:"
template_test['target'] = "{}"