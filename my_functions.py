import dionysus as d # persistence homology library
import numpy as np
import matplotlib.pyplot as plt
import music21 as m21 # music library


# change the original function in dionysus because there were problem with points at infinity
def drawDgm(D, boundary=None, epsilon=.5, color=None):
    '''
    Draws simple persistence diagram plot

    :param D:
        a persistence diagram, given as a Nx2 numpy array
    :param boundary:
        Boundary of persistence diagram axes
    :param epsilon:
        The diagram is drawn on [0,boundary]x[0,boundary].
        If boundary not given, then it is determined to be the
        max death time from the input diagram plus epsilon.
    :returns: Figure that includes persistence diagram
    '''
    # Separate out the infinite classes if they exist
    includesInfPts = np.inf in D
    if includesInfPts:
        Dinf = D[np.isinf(D[:, 1]), :]
        D = D[np.isfinite(D[:, 1]), :]

    # Get the max birth/death time if it's not already specified
    if not boundary:
        if (len(D)>0):
            boundary = D.max()+epsilon
        else:
            if (max(x for x in Dinf[:, 0])==0):
                boundary = 1
            else:
                boundary = max(x for x in Dinf[:, 0])*5/4

    # if fig is None:
    #     fig = plt.figure()
    # ax = fig.gca()
    # Plot the diagonal
    plt.plot([0, boundary], [0, boundary])

    # Plot the diagram points
    if color is None:
        plt.scatter(D[:, 0], D[:, 1])
    else:
        plt.scatter(D[:, 0], D[:, 1], c=color)

    if includesInfPts:
        #print(Dinf[:, 0])
        plt.scatter(Dinf[:, 0],[.98*boundary for i in Dinf[:, 0]], marker='s', color='red')
        #plt.scatter(Dinf[:, 0], .98*boundary, marker='s', color='red')
        plt.axis([-.01*boundary, boundary, -.01*boundary, boundary])

    plt.ylabel('Death')
    plt.xlabel('Birth')
    

def drawAll(d):
    '''
    Print all diagrams d
    '''
    for i, dmg in enumerate(d):
        for pt in dmg:
            print(f"{i}-dim diagram: {pt.birth},{pt.death}")
        
    fig, axes = plt.subplots(nrows=1, ncols=len(d), figsize=(6*len(d),5))
    for i, dgm in enumerate(d):
        if len(d)==1:
            plt.sca(axes)
            plt.title(str(i)+'-dim diagram')
            dgm_array = np.array([[pt.birth for pt in dgm],[pt.death for pt in dgm]]).T
            if [i for i in dgm_array] != []:
                drawDgm(dgm_array, color = 'black')
        else:        
            plt.sca(axes[i])
            plt.title(str(i)+'-dim diagram')
            dgm_array = np.array([[pt.birth for pt in dgm],[pt.death for pt in dgm]]).T
            if [i for i in dgm_array] != []:
                drawDgm(dgm_array, color = 'black')
    plt.show()

def C(n_1,n_2,n_3):
    '''
    Tonnetz's definition
    '''
    T = []
    N = n_1 + n_2 + n_3
    
    # add 0-simplices
    T = [[i] for i in range(N)]

    # add 1-simplices
    T += [[i,(i+k)%N] for i in range(N) for k in [n_1,n_2,n_3]]

    # add 2-simplices
    T += [[i,(i+n_1)%N, (i+n_1+n_2)%N] for i in range(N)]
    T += [[i,(i-n_1)%N, (i-n_1-n_2)%N] for i in range(N)]
        
    return T

def get_duration(score):
    '''
    Given a score in music21.stream format returns the distribution of the notes
    '''
    h = [0]*12
    for thisNote in score.flatten().getElementsByClass('Note'): # get all the notes in score
        h[thisNote.pitch.pitchClass] += thisNote.duration.quarterLength
    
    for thisChord in score.recurse().getElementsByClass('Chord'): # get all the chords in score
        for thisNote in thisChord:
            h[thisNote.pitch.pitchClass] += thisNote.duration.quarterLength
    
    # Eventually one could split the duration of the chord in the note used:
    #     h[thisNote.pitch.pitchClass] += thisNote.duration.quarterLength/thisChord.duration.quarterLength 
    return h

def pitch_class_distribution(h):
    '''
    Plot the distrubution h of the notes
    '''
    notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    plt.bar(notes,h,align="center",ec = "grey",fc="dodgerblue")
    plt.ylabel('duration')
    plt.xlabel('pitch class')

def per_diagms(h, T, norm = False):
    '''
    Given 
    h = distrubution of the notes
    T = Generalized Tonnetze (N=12)
    returns the persistence diagrams computed usind dionysus library
    '''
    
    if norm:
        #rinormalization
        h = [a/sum(h) for a in h]
    
    #create the "lower_star" filtration
    filtr = d.Filtration()
    for sigma in T:
        filtr.append(d.Simplex(sigma,max([h[i] for i in sigma])))
    filtr.sort()
    
    #computing homology persistence
    p = d.homology_persistence(filtr)
    dgms = d.init_diagrams(p, filtr)
    return dgms

def per_diagms_score (score, T, norm = True, print_h = False):
    h = get_duration(score)
          
    if print_h:
        print(f"pitch class distribution: {h}")
    
    return per_diagms(h, T, norm)


def bottle_dist_print(score1,score2,T,dim=2):
    '''
    Print of the bottleneck distance between the i-th diagram of score1 and the i-th diagram of score2 for every dimension i up to dim
    '''
    for i in range(dim+1):
        print(f"{i}-dim diagrams distance : {d.bottleneck_distance(per_diagms_score(score1,T)[i], per_diagms_score(score2,T)[i])}")
        
def transpose(s, inter):
    '''
    Transpose a score by an interval=inter
    '''
    for n in s.flatten().getElementsByClass('Note'):
        n.transpose(inter, inPlace=True)
    
    for thisChord in s.recurse().getElementsByClass('Chord'):
        for n in thisChord:
            n.transpose(inter, inPlace=True)
    return s

def bottle_distance_matrix(h_scores, k_hom, T):
    '''
    k_hom = [0] or [0,1] dimensions
    
    Compute the symmetric matrix bd[i] given by the pairwise distance of the i-th persistence diagrams with Tonnetz=T of the distributions h_scores
    '''
    n = len(h_scores) # numeber of scores
    bd = np.zeros((len(k_hom),n,n))
    for i in range(0,n):
        tmpi = per_diagms(h_scores[i], T, norm=1)
        for j in range(i+1, n):
            tmpj = per_diagms(h_scores[j], T, norm=1)
            for k in k_hom:
                bd[k][i][j] = d.bottleneck_distance(tmpi[k], tmpj[k])
                bd[k][j][i] = bd[k][i][j]
    return bd

def dendrogram_plot(distance_matrix, score_names, testo, linkage_type = 'average',save=0):
    '''
    Print hierarchical clustering with linkage_type given a dissimilarity matrix distance_matrix
    '''
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    dist = squareform(distance_matrix)
    linkage_matrix = linkage(dist, linkage_type)
    dendrogram(linkage_matrix,orientation='right', labels=score_names)
    plt.title(testo)
    if save:
        plt.savefig(testo+'.png',format='png',bbox_inches='tight',pad_inches=0.5)
    plt.show()
  
Torus_tonnetz = [(C(3,4,5),"C(3,4,5)"),
                 (C(1,2,9),"C(1,2,9)"),
                 (C(1,3,8),"C(1,3,8)"), 
                 (C(1,4,7),"C(1,4,7)"),
                 (C(2,3,7),"C(2,3,7)")]
