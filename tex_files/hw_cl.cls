\LoadClass[10pt,letterpaper]{article}
\ProvidesClass{hw_cl}

%%%% PACKAGES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Layout & Formatting
\usepackage{lmodern}
\usepackage[utf8]{inputenc}
\usepackage[margin=1.0in]{geometry}		% For Margins
\usepackage{fullpage}
\usepackage{framed,xcolor} 				% Framed Paragraph Boxes
\usepackage{enumitem}					% Customizable Lists
\usepackage{parskip}



% Fonts
\usepackage{mathrsfs}                   % Ralph Smith's Formal Script Symbol Fonts 
\usepackage{marvosym}
\usepackage{amsfonts}                   % AMS Fonts
\usepackage{bm}

% Figures / Pictures
\usepackage{graphicx,subcaption}		% Image Embedding
\usepackage{float}						% Floats
\usepackage{subcaption}					% Subfigures
\usepackage[margin=1cm]{caption}		% Caption Formatting
\usepackage{array}                      % Arrays
\usepackage{multirow}                   % Merge Table Rows
\usepackage{booktabs}                   % Pretty Tables
\usepackage{listings}                   % Verbatim Listings
\usepackage[ruled]{algorithm2e}         % Pseudocode
\usepackage{verbatim}

% Math / Theorems
\usepackage{amsmath,amsthm,amssymb}		% AMS Math
\usepackage{thmtools}					% Theorem Tools
\usepackage{mathtools}					% Theorem Tools
\usepackage{esint}						% Alternate Integral Symbols

% Misc
\usepackage{url}						% URL Formatting
\usepackage{soul}                       % Sane Unnderlines
\usepackage{color}                      % Colors
\usepackage{fancyhdr}

%%%% COMMANDS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\newcounter{hwcount}
\numberwithin{equation}{hwcount}

% Sets
\newcommand{\set}[1]{\{#1\}}

% Blackboard Bold
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\E}{\mathbb{E}}
% Calligraphic
\newcommand{\A}{\mathcal{A}}
\newcommand{\B}{\mathcal{B}}
\newcommand{\D}{\mathcal{D}}
\newcommand{\F}{\mathcal{F}}
\renewcommand{\L}{\mathcal{L}}
\newcommand{\M}{\mathcal{M}}
\renewcommand{\P}{\mathcal{P}}
\newcommand{\X}{\mathcal{X}}
\newcommand{\Y}{\mathcal{Y}}

% Symbols
\newcommand{\eps}{\varepsilon}
\newcommand{\abs}[1]{|#1|}
\newcommand{\suchthat}{\text{ s.t. }}
\newcommand{\pipe}{\;|\;}
\renewcommand{\vec}[1]{\boldsymbol{#1}}

% Calculus
\newcommand{\diff}[1]{\,\text{d}#1}
\newcommand{\pdiff}[2]{\frac{\partial#1}{\partial#2}}

% Linear Algebra
\DeclareMathOperator{\im}{Im}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\nullity}{nullity}
\DeclareMathOperator{\adj}{adj}
\DeclareMathOperator{\trace}{tr}
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\rref}{rref}
\DeclareMathOperator{\dist}{dist}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\Cov}{Cov}
\DeclarePairedDelimiter{\norm}{\lVert}{\rVert}
\DeclarePairedDelimiter{\inner}{\langle}{\rangle}
\DeclarePairedDelimiter{\sigalg}{\langle}{\rangle_\sigma}
\DeclarePairedDelimiter{\lambsys}{\langle}{\rangle_\lambda}

% Fourier Transforms
\newcommand{\fhat}{\hat{f}}

% Distributions
\newcommand{\Ber}{\text{Ber}}
\newcommand{\Bin}{\text{Bin}}
\newcommand{\Geom}{\text{Geom}}
\newcommand{\Dir}{\text{Dir}}
\newcommand{\Beta}{\text{Beta}}

%%%% HOMEWORK INFO / STRUCTURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{datetime}
\usdate
\settimeformat{ampmtime}

% Title Heading
\renewcommand{\maketitle}{
	\begin{center}
	    \newlength{\titlerulewidth}
	    \def \hmwkttl{{\Large \textbf{\TVclass\, - \TVassignment}}}
	    \settowidth{\titlerulewidth}{\hmwkttl}
        
	    \rule{\titlerulewidth}{1pt}\\[3mm]
	    \hmwkttl \\[3mm]
	    \makebox[\titlerulewidth]{\small \TVname \hspace{1em} \hfill \hfill  Due: \TVdue} \\
	    \makebox[\titlerulewidth]{\small Edited by: \TVedit \hspace{1em} \hfill \hfill  Last Updated: \today \ \currenttime} \\
	    \rule{\titlerulewidth}{1pt}\\[3mm]
	\end{center}
	\vspace{3em}
}

\usepackage{titlesec}
\titleformat{\section}
  {\LARGE\bfseries}{\thesection. }{0em}{}

% Document Info
\makeatletter
\def \TVname{}
\newcommand{\hwclass}[1]{\def \TVclass{#1}}
\def \TVedit{}
\newcommand{\hwdue}[1]{\def \TVdue{#1}}
\newcommand{\hwassignment}[1]{\def \TVassignment{#1}}
\makeatother

% Lengths
\setlength{\headsep}{25pt}
\setlength{\voffset}{-25pt}
\setlength{\headheight}{12pt}
\setlength{\parskip}{0.5em}

% Custom problem counters
\newcounter{probnum}
\newcounter{subprobnum}
\newcounter{subsubprobnum}
\stepcounter{probnum}
\stepcounter{subprobnum}
\stepcounter{subsubprobnum}

%%%% CUSTOM ENVIRONMENTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Proof List
\newlist{prooflist}{enumerate}{3}
\setlist[prooflist]{noitemsep, leftmargin=2em}
\setlist[prooflist,1]{label=(\arabic*)}
\setlist[prooflist,2]{label=(\alph*), topsep=0pt}
\setlist[prooflist,3]{label=(\roman*), topsep=0pt}
\setlist[prooflist,4]{label=(\arabic*), topsep=0pt}

\setlength{\topsep}{1em}
%\theoremstyle{definition}
\newtheoremstyle{problems}
  {\topsep}   % above space
  {\topsep}   % below space
  {}  % body font
  {0pt}       % indent
  {\bfseries} % head font
  {.}         % head punctuation
  {5pt} % HEADSPACE
  {}          % CUSTOM-HEAD-SPEC
\setcounter{probnum}{0}
\theoremstyle{problems}
\newtheorem{problem}[probnum]{Problem}

%% Problem Environment

\usepackage{substr}

\newtheoremstyle{subproblems}
  {.5em}   % above space
  {.5em}   % below space
  {}  % body font
  {1em}       % indent
  {} % head font
  {.}         % head punctuation
  {5pt plus 1pt minus 1pt} % HEADSPACE
  {\underline{\thmname{#1}{\thmnumber{ #2}}}}          % CUSTOM-HEAD-SPEC

\theoremstyle{subproblems}
\newtheorem{subproblem}{Part}[problem]

\renewcommand{\thesubproblem}{\Alph{subproblem}}

%% Named Environment
\newenvironment{named}[2]
	{\begin{center}{\textbf{#1}}\end{center}\begin{trivlist}
	    \item[\hskip \labelsep (#2)]}
	{\end{trivlist}}

%% Underlined Section
\newenvironment{underlined}[1]
	{\begin{trivlist}
		\item[\hskip \labelsep {\underline{#1.}}]}
	{\end{trivlist}}

%% Answer
\newenvironment{answer}
	{\begin{trivlist}\item}
	{\end{trivlist}}