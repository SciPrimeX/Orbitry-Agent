# Glucoza: Advisor Agent

## 1. Idea and Business Value

An agent that handles *glucose- and diabetes-related advice* for users leveraging knowledge from four sources: 1) the *history* of interactions and medical history of the user if provided; 2) *SPXFusion* analytics for this patient, i.e., most recent results from the ML+DL ensemble forecasting glucose/predicting hypoglycemia (HG) risk; 3) *SPXFusion* population analytics, i.e., drawing comparisons to public data as well as anonymized data and model results from other users (larger user base => more accurate advice); 4) an updated *corpus of research* on the connection between the non-invasive features in SPXFusion and glucose, including restrictions on recommended insulin levels, enforcement of correct boundaries for low and high glucose levels, interpretation of ECG in relation to glucose, etc.

<img src='https://i.ibb.co/C5RfP1nX/Glucoza.png' width=1200>

The Agent refers to a multi-agentic pipeline (MAP) presented in Figure 1. The MAP receives a signal from the environment through the API: either SPXFusion detects a high risk of hypoglycemia (focus only on HG right now for clarity), or a User chat indicates directly or indirectly (e.g., the LLM can detect if the person is feeling symptoms without connecting them to HG) that the User is about to experience HG. These are two structurally different signals, but handled through one MAP for future scalability (see Section 2). The MAP is an *orchestrated system of multiple agents* (with potential sub-agents):

1. **Coordinator**: receives the signal, delegates to the correct worker, evaluates workers' outputs before passing to another worker. *Brain*: orchestration-tuned LLM_1. *Tools*: None. *Trained* to coordinate and validate results from agents before passing them over.

2. **Data Analyst** (worker 1): receives a structured query from the Coordinator, analyzes SPXFusion data for the User and in general, returns statistical tests, p-values, charts, etc. *Brain*: LLM_2 tuned for analyzing the ML ensemble *SPXFusion*, i.e., testing that the user is actually in a high-risk zone with distribution tests like Mann-Whitney, running simulations and A/B tests (e.g., what if the user eats a sugary snack?) using *SPXFusion*, drawing charts for the research analyst. *Tools*: Python to run *SPXFusion*, perform stat tests, and plot charts. *Trained* to produce the most relevant tests and graphs.

3. **Research Analyst** (worker 2): receives a structured query from the Coordinator (potentially including validated results from the Data Analyst), retrieves relevant scientific knowledge with *dynamic and personalized RAG*, continuously updated with new knowledge to stay up to standards, returns a short summary related to the initial signal including references to sources and the reasoning trace (for observability). *Brain*: a fairly plain LLM_3 for summarizing documents, as the power of this agent is in the design of the knowledge system. *Tools*: WebSearch to complement retrieval or update the knowledge db. *Trained* for fast and accurate knowledge handling.

4. **Communicator** (worker 3): receives a structured query from the Coordinator (potentially including results from Data and Research Analysts), uses an extensively fine-tuned LLM engine for precise, sensitive, accurate diabetes-related advice, and generates the text output that will face the user. *Brain*: ideally a scratch-made or heavily tuned LLM_4 that generates coherent recommendations. *Tools*: None, just generating output "Good Advice". *Trained* to maximize user trust + adoption of recommendations + actual wellbeing if data is available.

As a result, the Agent returns a user-friendly text recommendation backed by user- and model-data as well as scientific research while staying friendly — **"Good" Advice**. The quotation marks around "good" are not to say that it is bad; they are just to emphasize the relativity of what "good" means — it will need its own metrics, approaches to evaluation, and validation. Note that this system can respond to signals that do not require research, as it is up to the Controller to activate analysts or not. Hence, a simple chat can be implemented with that Agent as well.

**Business value** of Glucoza is rooted in the following facts:

* Users might not want to go to the doctor with their concern while it can be serious;
* Timely intervention has high potential for increasing user wellbeing;
* Personalization that accounts for user-specific knowledge keeps them with our product longer.

This way we fill a big gap for people suffering from volatile glucose: a low-cost solution to daily anxieties about their well-being. Something neither CGM nor Apple or Samsung are able to do.

Of course, many details will be clarified to differentiate between advising for HG, glucose, Type-1 and Type-2 diabetes, as well as the general public or general glucose screening (like athletes). Reinforcement Learning is core to all training (see [TRL](https://github.com/huggingface/trl)): making algorithms maximize explicit and auditable reward functions as they interact with user data helps retain and increase the user base, which in turn increases the quality of Glucoza. LLM and Agent evaluation is key as well and is automated within the Coordinator (potentially separated into its own sub-agent). All of this is absolutely top-notch in agentic systems.

## 2. Scalability

As SciPrimeX is positioned right now, it needs to scale well to more general wellness advice. Scalability can be sketched in order of increasing generalization:

1. *Type 1 Diabetic --> Type 2 Diabetic --> General population*. The first transition is possible via fine-tuning and semi-supervised learning since data for T2D glucose exists, but it's sparse. An RL model can create a synthetic benchmark to maximize accuracy on existing T1D glucose data as well as physiological plausibility in relation to non-invasive features. This transition is recently researched and proven to work well. Transitioning to the general population is more about transfer learning, as we can't train the model explicitly to give glucose-related advice to people we don't have glucose data for. However, there is a good chance the Agent will generalize well since the SPXFusion predictive model will also adjust to the new covariate distribution and extrapolate. Hopefully, they cancel each other's errors rather than multiply them.

2. *Strictly glucose-related intervention --> diabetes advice --> wellness advice*. Along the same lines, but extrapolation will be in the tasks that the Agent performs. There are two ways to look at it, as the Agent is able to give all three types of advice at any stage since we are using LLMs pre-trained on comprehensive corpuses. However, scientific plausibility as well as hallucinations are very likely. Hence, this type of scalability implies redefining and expanding tasks managed by the agent as well as the knowledge bases.

3. *Glucose --> Glucose-related physiology --> Non-glucose physiology*. By using the same non-invasive features, we can expand (using pretty much the same mechanism) to physiological patterns that are either derivative from glucose or derivative from other latent factors that cannot be measured non-invasively or are hard to measure without going to the doctor (like arrhythmia). At this stage, the possibilities are limitless, as we can include all things interesting to the wellness-curious user such as sleep analysis (integrating with existing apps), activity suggestions, dietary suggestions, you name it. With that, we can reach a really wide user base while staying grounded in research and strict physiological predictions. This will require automating knowledge base updates as well as the development of other predictive ML ensembles.

4. *Multi-modality*: storing objects that are not text, like images of ECG, in the knowledge db --> interacting with the user through images and voice.

## 3. FFplus

Funding opportunity aimed at empowering SMEs for "innovation studies" with Generative AI. As the whole agentic system internally is based on LLM-generated queries to each worker and the final user-facing output is a type of conversation, this project clearly fits the topic. Moreover, FFplus is short-termed (10 months starting from September 1st) and oriented towards scientifically backed improvements and GenAI R&D that uses high-performance computing (HPC). In our case, it means that developing LLMs (fine-tuning, creating new embedding systems, pre-training LLMs from scratch) most suited for our business plan is exactly what FFplus is for. It is not as much focused on delivering the product as on *improving the product using HPC*. Throughout Stage 1 of development (see Section 5), we will write a grant application and proposal rooted in the initial iteration of the Agent, i.e., what and how exactly we need to R&D to increase our value. My initial vision is that deeply tuning LLMs, building our own LLM engine for the Coordinator, improving reasoning, and creating automated agent-evaluation are key and set us at the very front of agentic enterprises.

## 4. Stack

* **Agents**: LangChain, LangSmith, CrewAI (for experiments on costs and latency), MCP, A2A (+ADK?), PyTorch/TF (depending on needs and costs), HuggingFace (LLMs for tuning)
* **Cloud**: GCS, Vertex Agent Builder, Cloud Run (API + app), Artifact Registry (training containers)
* **Monitoring**: TBD (monitor training + online performance + cloud load)
* **Front**: Streamlit (for now)

## 5. Timeline

#### Stage 1

* **Week 1. January 19-25:** initialize, team formation, onboarding, planning, initialize FFplus
* **Week 2. January 26-31:** Backend development of Glucoza v0, write FFplus
* **Week 3. February 2-8:** Deploy Glucoza v0, experiment on cloud costs for tuning and training LLMs, edit and submit FFplus
* **Week 4. February 9-15:** Evaluate Glucoza v0, simplest Streamlit front as blueprint

#### Stage 2

* **Weeks 5-8 by March 15:** Backend Glucoza v1 improvement, fine-tuning LLMs, RAG+knowledge graphs
* **Week 9 March 16-22:** Deploy and update frontend (ideally in a CI/CD manner by now)
* **Week 10 by April 1:** Testing chat signals with the team, testing handling model alarms on public data

#### Stage 3

* **Q2 of 2026:** beta-testing chat with first users --> getting first willingness to pay + testing handling model data with real-time updates from Galaxy Watch.
* **By September 1:** when potentially FFplus arrives, we have the product functioning already --> spend FFplus on R&D and immediately deploy improved Glucoza.

## 6. Goals

1. Repo, onboarding, FFplus init
2. Functioning agentic graph using off-the-shelf LLMs (like Gemini or DeepSeek)
3. Benchmark evaluation using appropriate metrics
4. Increase performance from benchmark by developing each worker and coordinator in parallel:
   1. Coordinator: separate quality control from it? fine-tune LLM or use fundamental models? needs assistants?
   2. Data Analyst: optimize for analyzing model results, not raw data
   3. Research Analyst: efficient and updating RAG and knowledge graphs
   4. Communicator: fine-tuning on medical corpus (cf. [HuggingFace](https://huggingface.co/ruslanmv/Medical-Llama3-8B))
5. Implement persistent memory for the Agent
6. Document the pipeline in FFplus

## 7. Team needs

Three engineers (Santanu, Abdati, myself) is more than enough to work on development. Potential subscription for Claude Code and [deeplearning.ai](http://deeplearning.ai). Cloud infrastructure is on John and myself. After the first frontend as a blueprint, it is delegated to Abraham and UX/UI. Legal team needs to find exact requirements for FAIR and data compliance. Market research across medical AI advisors that use agents/GenAI. Sustainability benchmarks from Alex (offsetting CO₂, ethical text interactions).

## 8. Cloud needs

The exact financial needs will be determined through early testing. So far, I will use my $2000 from Le Wagon + Abdati and Santanu can utilize theirs for parallel development. Should be enough to get through Q1 of 2026 and get to Stage 3.

#### Storage

1. Stage 1 only requires storing pre-trained LLM models.
2. Stage 2 will expand to (in decreasing order of size/cost) text corpuses for fine-tuning LLMs + medical research db + test chat history and metadata.

#### Computing

1. Stage 1 requires forward-pass throughput (to generate responses), experiment with costs for fine-tuning on small data and small jobs.
2. Stage 2 automated fine-tuning and potentially full training of multiple LLMs.

#### Serving

1. Stage 1 deploys the model and API, low request volume.
2. Stage 2 monitoring Agent performance, higher API volume, persistent memory.
