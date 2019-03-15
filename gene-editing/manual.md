# Manual for Use of geneedit.py

# Contents

- Introduction
  - Description and Features
  - Installation
  - License
  - Contributors
- Methodology
- Parameters used in **geneedit.py**
- References

# Introduction

## Description and Features

This manual describes how to use **geneedit.py**, a program for the simulation
of gene editing strategies applied to livestock populations.

The widespread adoption and corresponding reduction in the cost of high-density
single nucleotide polymorphism (SNP) genotyping has enabled the detection of
many new recessives that have deleterious effects on fertility in several
breeds of dairy cattle, and whole genome sequencing allows detecting additional
fertility defects. Many of these new recessives were not previously detected by
test matings because they cause embryonic losses in early gestation that could
not be distinguished from failed breedings. Annual losses to U.S. dairy farmers
from decreased fertility and increased perinatal mortality due to known recessive
defects are estimated to be at least $10 million (€9,370,754). Mate allocation
tools do not always consider carrier status when bull and cow pairs are assigned,
and few make use of DNA marker or haplotype information. Avoiding
carrier-to-carrier matings is easy when only a few recessives are segregating in
a population but is considerably more difficult when many defects are
segregating.

Cole recently extended a simple method for controlling the rate of increase in
genomic inbreeding proposed by Pryce et al. to account for economic losses
attributable to recessive defects. In the original method, parent averages
(PAs) for matings that produced inbred offspring were penalized, and the bull
that produced the highest PA after the inbreeding adjustment was selected in a
sequential manner. The number of matings permitted for each bull was constrained
to prevent one bull with high genetic merit from being mated to all cows. Cole
modified this approach to include an additional term that penalized
carrier-to-carrier matings that may produce affected embryos and showed that the
additional penalty decreased minor allele frequency (MAF) faster than other
methods. However, many generations of selection were still needed to eliminate
recessives from the population, and some defects remained in the population at
low frequency.

A number of tools are now available for editing eukaryotic genomes, including
clustered regularly interspaced short palindromic repeats (CRISPR),
transcription activator-like effector nucleases (TALEN), and zinc finger
nucleases (ZFN). Treating simple recessive disorders by using gene editing is
of great interest, and CRISPR has been used to generate pigs that are resistant
to porcine reproductive and respiratory syndrome. Gene editing also has been
used to produce desirable phenotypes (e.g., polled cattle). A recent series of
simulation studies showed that gene editing also has the potential to improve
rates of genetic gain for quantitative traits. Gene editing may be an effective
means of reducing the frequency of genetic disorders in livestock populations
or eliminating those disorders altogether.

Similar programs:
- AlphaSimR  

## Installation

The latest version of **geneedit.py** can be [downloaded from GitHub](https://github.com/wintermind/).

Requirements:
- Python 2.7
  - matplotlib (visualization)
  - import cerberus (parameter checking)
  - numpy (matrix operations)
  - pandas (reports and summaries) 
  - scipy (random variates)
- [inbupgf90](...) version 1.42 or later (inbreeding calculations)

The program has been developed and tested on Centos Linux and macOS 10.14.2
(Mojave). In principle, it should work on other platforms, such as Microsoft
Windows, but the authors have not verified this.

## License
**geneedit.py** was created by an employee of the United States government as
part of their official duties, and is therefore in the public domain. 

## Contributors
- Project creator: [John B. Cole](john.cole@ars.usda.gov)
- Contributor: [Maci L. Mueller](...) 

# Methodology

## Simulation
The simulation software of Cole [14] was modified to include four different gene editing technologies and used to examine several scenarios for the use of gene editing in a dairy cattle population. With the exception of the gene editing methodology, the simulation procedures were identical to those described in detail by Cole [5]. A base population of 350 bulls and 35,000 cows with ages ranging from 1 to 5 was simulated by drawing true breeding values (TBV) from normal distributions and randomly sampling genotypes from a Bernoulli distribution with a parameter equal to that locus’s major allele frequency. The distribution of bull TBV had a greater variance ( = 300) than did the cow TBV ( = 200), and both had a mean of 0. At least one carrier bull and cow were present in the base population for each recessive, and a constant mutation rate of 10-5 that resulted in “AA” genotypes being converted to “Aa” genotypes when calves were created was used to prevent all minor alleles from being lost due to drift. All base population animals were treated as founders and mated for 10 years to produce the population used for gene editing. Each round of the simulation represents one year of calendar time, and generations overlap. The population sized increased each round during the burn-in period until a maximum of 500 bulls and 100,000 cows were in the population. Each year, animals were culled for reaching the maximum permitted age (10 years for bulls and 5 years for cows), and then at random to maintain population size, if necessary. All simulated loci are assumed to be inherited independently of one another (no linkage disequilibrium is assumed). The method used to allocate bulls to cows for mating is described in detail below. All simulation parameters used in the simulations are shown and discussed in detail in Appendix 1.
Table 1 Simulation parameters
Software parameter	Definition	Value
base_bulls	Number of bulls in the base population	350
base_cows	Number of cows in the base population	35,000
service_bulls	Number of bulls in the sire portfolio used by each herd	50
base_herds	Number of pseudo-herds used in the simulation	200
max_bulls	Maximum number of bulls available for use as service sires in each generation	500
max_cows	Maximum number of cows in the population in each generation	100,000
generations	Number of generations simulated	20
max_matings	Maximum number of matings each service sire is permitted each year	5000
debug	Show or hide debugging messages	True
history_freq	Frequency with which history files are saved to disk	End
rng_seed2	Value used to seed the random number generator	Time + PID
edit_prop	Proportions of bulls and cows edited in different scenarios	0%, 1%, 10% (bulls); 
0%, 1% (cows)
edit_type3	Technologies used for gene editing	C, P, T, Z
Time system clock time when the simulation is submitted, PID process identification reported by the operating system, C clustered regularly interspaced short palindromic repeats, T transcription activator-like effector nuclease, P hypothetical technology with perfect success rate, Z zinc finger nuclease

Mate allocation
The modified Pryce scheme accounting for recessive alleles described by Cole [5] was used to allocate bulls to cows in all scenarios. The selection criterion was the 2014 revision of the lifetime net merit (NM$) genetic-economic index used in the United States [15]. For each herd, 20% of the bulls were randomly selected from a list of live bulls, and the top 50 bulls from that group were selected for use as herd sires based on true breeding value (TBV). This produced different sire portfolios for each herd and is similar to the approach of Pryce et al. [6].
As in Cole [5], a matrix of PAs ( ) was constructed with rows corresponding to bulls and columns corresponding to cows as

B_ij^'=0.5(TBV_i+TBV_j )-λF_ij-∑_(r=1)^n▒〖P(aa)_r×v_r 〗,
where   is the PA for offspring of bull i and cow j,   is the TBV NM$ for bull i,   is the TBV NM$ for cow j, λ is the inbreeding depression in dollars associated with a 1% increase in inbreeding,   is the pedigree coefficient of inbreeding (%) of the calf resulting from mating bull i to cow j, n is the number of recessive alleles in a scenario,   is the probability of producing an affected calf for recessive locus r, and   is the economic value of locus r. The regression coefficient of NM$ on inbreeding (λ) was computed as the weighted average of the December 2014 effects of inbreeding on the traits in the index as done by Cole [5]; the weights correspond to those assigned to each trait in the NM$ index and resulted in a λ of $25. The P(aa) equals 0.25 for a mating of two carriers, 0.5 for a mating of an affected animal with a carrier, or 1 for a mating of two affected animals. Twelve recessive loci were used in the simulations (Table 2).
Table 2 Properties of the recessive loci included in each simulated scenario
Frequency	Value ($)a	Name
Lethal
0.0276	150	Brachyspina	Yes
0.0192	40	HH1	Yes
0.0166	40	HH2	Yes
0.0295	40	HH3	Yes
0.0037	40	HH4	Yes
0.0222	40	HH5	Yes
0.0025	150	BLAD	Yes
0.0137	70	CVM	Yes
0.0001	40	DUMPS	Yes
0.0007	150	Mulefoot	Yes
0.9929	40	Horned	No
0.0542	20	Red coat color	No
HH1, HH2, HH3, HH4, HH5 Holstein fertility haplotypes 1,2,3,4,5, respectively, BLAD bovine leukocyte adhesion deficiency, CVM complex vertebral malformation, DUMPS deficiency of uridine monophosphate synthase
aPositive values are undesirable and negative values are desirable. Values differ across loci because timing of pregnancy loss or calf death varies from one locus to another [1]. 
After   was constructed, a matrix of matings (M) was used to allocate bulls to cows. An element ( ) was set to 1 if the corresponding   value was the greatest value in column j (that bull produces the largest PA of any bull available for mating to that cow); all the other elements of that column were set to 0. If the sum of the elements of row i was less than the maximum number of permitted matings for that bull, then the mating was allocated. Otherwise, the bull with the next-highest   value in the column was selected. This procedure was repeated until each column had only one element equal to 1.
Gene editing
The model of gene editing in the simulation is based on the use of somatic cell nuclear transfer [2,3] to produce clones of high-genetic-merit animals that are then edited, and is presented schematically in Figure 1. The original (donor) animals remain in the population, and their clones may enter the population if the gene editing and embryo transfer steps are successful. The following were repeated for each locus to be edited: 
Step 1: Sort candidates for editing on TBV in descending order.
Step 2: Select animals to be edited using the user-specified proportion. For example, if 1% (0.01) of 500 bulls are to be edited, clones are created of the 5 bulls with the best (largest) TBV.
Step 3: New animals representing the clones of the bulls and cows to be edited are created and added to the list of animals. The original animals are not affected by the cloning process and remain in the population. All clones of bulls are males, and all clones of cows are females.
Step 4: The gene editing process is repeated for each locus to be modified, and each editing process is assumed to be independent (a success or failure for one locus has no effect on other loci). If only a fixed number of trials are permitted for gene editing, a vector of Bernoulli variates of length equal to the permitted number of trials is drawn and searched for the first (earliest) success. Otherwise, the editing process is repeated until a successful outcome is observed for that locus. The number of trials needed to achieve a successful edit is stored in the animal record. If this process is successful, Aa and aa genotypes of the clone are edited to AA genotypes (all edited animals are assumed to be homozygous for the major allele).
Step 5: If only a fixed number of trials are permitted for the embryo transfer step, a vector of Bernoulli variates of length equal to the permitted number of trials is drawn and searched for the first (earliest) success. Otherwise, the embryo transfer process is repeated until a successful outcome is observed (a live calf is born). The number of trials needed to achieve a successful live birth is stored in the animal record. If this process is successful, a live calf with an edited genotype is added to the live animal list. If this process is unsuccessful, the calf is assigned to the dead animals list.
Figure 1 shows a flowchart describing the key steps in the gene editing and embryo transfer processes.
Please place Fig. 1 around here
Three laboratory approaches to gene editing (CRISPR, TALEN, and ZFN) were supported as well as a fourth method that assumes that editing always is successful. The CRISPR, TALEN, and ZFN methods differed in their editing success and embryonic death rates [7,8], so the number of trials needed for a 99% probability of producing a live, gene-edited calf were calculated using the inverse cumulative probability function of a geometric distribution with parameters 0.99 and editing success rate  embryo transfer success rate as implemented in the scipy.stats.geom.ppf function of SciPy version 0.19.1 (Table 3). Bulls and cows can be edited at different rates (e.g., 10% of bulls and 1% of cows), or not edited at all by setting the rate for a sex to 0. Any combination of loci can be edited, and the number of edited loci was not restricted. A scenario in which no genes were edited, which reflects current practice, was used as the baseline against which the various editing scenarios were compared.
Table 3 Gene editing and embryo transfer success rates and trials needed for a live calf
Technology	Editing success rate	Embryo transfer success rate	Success probabilitya	Trials (no.)
				Successful edit	Successful ET	Live calf
CRISPR	0.63	0.21	0.13	5	20	100
TALEN	0.21	0.12	0.03	20	37	740
Perfect	1.00	1.00	1.00	1	1	1
ZFN	0.11	0.08	0.01	40	56	2240
CRISPR clustered regularly interspaced short palindromic repeats, TALEN transcription activator-like effector nuclease, Perfect hypothetical technology with a perfect success rate, ZFN zinc finger nuclease, ET embryo transfer.
aCalculated as editing success rate  embryo transfer success rate.


# References

- Cole, J.B., and M.L. Mueller. 2019. Management of Mendelian traits in
  breeding programs by gene editing: A simulation study. (Under revision.)
  Preprint on bioRxiv: https://doi.org/10.1101/116459. 
- Mueller, M., J. Cole, T. Sonstegard, and A. Van Eenennaam. 2019. Comparison
  of gene editing vs. conventional breeding to introgress the POLLED allele
  into the U.S. dairy cattle population. J. Dairy Sci. (In press.) 
  https://doi.org/10.3168/jds.2018-15892.