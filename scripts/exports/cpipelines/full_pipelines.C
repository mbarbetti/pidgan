#define FLOAT_T float
// #define DEBUG 1

#ifndef BUFFERSIZE
#define BUFFERSIZE 1024
#endif 

#ifndef N_INPUT_RICH
#define N_INPUT_RICH 4
#endif 

#ifndef N_INPUT_MUON
#define N_INPUT_MUON 4
#endif 

#ifndef N_INPUT_GLOBALPID
#define N_INPUT_GLOBALPID 11
#endif 

#ifndef N_INPUT_GLOBALMUONID
#define N_INPUT_GLOBALMUONID 10
#endif 

#ifndef N_OUTPUT_RICH
#define N_OUTPUT_RICH 4
#endif 

#ifndef N_OUTPUT_MUON
#define N_OUTPUT_MUON 2
#endif 

#ifndef N_OUTPUT_GLOBALPID
#define N_OUTPUT_GLOBALPID 7
#endif 

#ifndef N_OUTPUT_GLOBALMUONID
#define N_OUTPUT_GLOBALMUONID 2
#endif 

#ifndef N_RANDOM_RICH
#define N_RANDOM_RICH 64
#endif 

#ifndef N_RANDOM_MUON
#define N_RANDOM_MUON 64
#endif 

#ifndef N_RANDOM_GLOBALPID
#define N_RANDOM_GLOBALPID 64
#endif 

#ifndef N_RANDOM_GLOBALMUONID
#define N_RANDOM_GLOBALMUONID 64
#endif 

#ifndef N_OUTPUT
#define N_OUTPUT 15
#endif 

#ifndef MUON_ERRORCODE
#define MUON_ERRORCODE -1000
#endif

#ifdef DEBUG
#include <stdlib.h>
#include <stdio.h>
#endif 

extern "C" FLOAT_T* Rich_muon_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Rich_muon_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Rich_muon_dnn                   (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* Rich_pion_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Rich_pion_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Rich_pion_dnn                   (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* Rich_kaon_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Rich_kaon_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Rich_kaon_dnn                   (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* Rich_proton_tX                  (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Rich_proton_tY_inverse          (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Rich_proton_dnn                 (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* Muon_muon_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Muon_muon_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Muon_muon_dnn                   (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* Muon_pion_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Muon_pion_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Muon_pion_dnn                   (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* Muon_kaon_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Muon_kaon_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Muon_kaon_dnn                   (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* Muon_proton_tX                  (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Muon_proton_tY_inverse          (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* Muon_proton_dnn                 (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalPID_muon_tX               (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPID_muon_tY_inverse       (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPID_muon_dnn              (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalPID_pion_tX               (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPID_pion_tY_inverse       (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPID_pion_dnn              (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalPID_kaon_tX               (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPID_kaon_tY_inverse       (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPID_kaon_dnn              (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalPID_proton_tX             (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPID_proton_tY_inverse     (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPID_proton_dnn            (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalMuonId_muon_tX            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonId_muon_tY_inverse    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonId_muon_dnn           (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalMuonId_pion_tX            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonId_pion_tY_inverse    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonId_pion_dnn           (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalMuonId_kaon_tX            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonId_kaon_tY_inverse    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonId_kaon_dnn           (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalMuonId_proton_tX          (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonId_proton_tY_inverse  (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonId_proton_dnn         (FLOAT_T *, const FLOAT_T *);

typedef FLOAT_T* (*mlfun) (FLOAT_T *, const FLOAT_T *); 


extern "C"
FLOAT_T* gan_pipe ( mlfun tX               , 
                    mlfun model            , 
                    mlfun tY               , 
                    FLOAT_T *output        , 
                    const FLOAT_T *input   , 
                    const FLOAT_T *random  , 
                    unsigned short nIn     , 
                    unsigned short nOut    , 
                    unsigned short nRandom )
{
    unsigned short i; 
    FLOAT_T buf_input[BUFFERSIZE]; 
    FLOAT_T buf_output[BUFFERSIZE]; 

    tX (buf_input, input);

    #ifdef DEBUG
    printf("Preprocessed input\n");
    for (i = 0; i < nIn; ++i)
        printf("in[%d] -> pin[%d] : %.2f -> %.2f\n", i, i, input[i], buf_input[i]); 
    #endif

    for (i = 0; i < nRandom; ++i)
        buf_input[nIn + i] = random[i]; 

    model (buf_output, buf_input);

    tY (output, buf_output); 

    #ifdef DEBUG
    printf("Preprocessed output\n");
    for (i = 0; i < nOut; ++i)
        printf("pout[%d] -> out[%d] : %.2f -> %.2f\n", i, i, buf_output[i], output[i]); 
    #endif 

    return output; 
}


extern "C"
FLOAT_T* GenericPipe ( FLOAT_T* output, const FLOAT_T *input, const FLOAT_T *random ,
                       mlfun richTx,  mlfun richModel,  mlfun richTy  , 
                       mlfun muonTx,  mlfun muonModel,  mlfun muonTy  , 
                       mlfun gpidTx,  mlfun gpidModel,  mlfun gpidTy  , 
                       mlfun gmuidTx, mlfun gmuidModel, mlfun gmuidTy )
{
    short i, j;
    float isMuon;

    // Split the random array into four sub arrays
    j = 0; 
    const FLOAT_T* r0 = random + j;
    j += N_RANDOM_RICH; 
    const FLOAT_T* r1 = random + j;
    j += N_RANDOM_MUON; 
    const FLOAT_T* r2 = random + j;
    j += N_RANDOM_GLOBALPID; 
    const FLOAT_T* r3 = random + j;

    // Rich
    FLOAT_T richinput [N_INPUT_RICH];
    FLOAT_T richdll [N_OUTPUT_RICH];
    for (i = 0; i < N_INPUT_RICH; ++i)
        richinput[i] = input[i]; 

    #ifdef DEBUG
    printf (" === RICH === \n");
    for (i = 0; i < N_INPUT_RICH; ++i)
        printf("RICH INPUT TRANSFER [%d] : %.2f -> %.2f\n", i, input[i], richinput[i]); 
    #endif

    gan_pipe ( richTx        ,
               richModel     ,
               richTy        ,
               richdll       ,
               richinput     ,
               r0            ,
               N_INPUT_RICH  ,
               N_OUTPUT_RICH ,
               N_RANDOM_RICH ); 

    // Muon
    FLOAT_T muoninput [N_INPUT_MUON];
    FLOAT_T muondll [N_OUTPUT_MUON];
    for (i = 0; i < N_INPUT_MUON; ++i)
        muoninput[i] = input[i]; 

    #ifdef DEBUG
    printf (" === MUON === \n");
    #endif
    isMuon = input[N_INPUT_RICH];

    if (isMuon > 0.5){
        gan_pipe ( muonTx        ,
                   muonModel     ,
                   muonTy        ,
                   muondll       ,
                   muoninput     ,
                   r1            ,
                   N_INPUT_RICH  ,
                   N_OUTPUT_RICH ,
                   N_RANDOM_RICH );
    }
    else
        for (i = 0; i < N_OUTPUT_MUON; ++i)
            muondll[i] = MUON_ERRORCODE; 

    // Global PID
    FLOAT_T gpid_input [N_INPUT_GLOBALPID];
    FLOAT_T gpid_output [N_OUTPUT_GLOBALPID];
    j = 0;

    // p, eta, nTracks, charge
    for (i = 0; i < N_INPUT_RICH; ++i)
        gpid_input[j++] = input[i]; 
    
    // dlle, dllmu, dllk, dllp
    for (i = 0; i < N_OUTPUT_RICH; ++i)
        gpid_input[j++] = richdll[i]; 

    // isMuon 
    gpid_input[j++] = isMuon; 

    // mullmu, mullbg
    for (i = 0; i < N_OUTPUT_MUON; ++i)
        gpid_input[j++] = muondll[i];

    #ifdef DEBUG
    printf (" === GLOBAL PID === \n");
    #endif
    gan_pipe ( gpidTx             ,
               gpidModel          ,
               gpidTy             ,
               gpid_output        ,
               gpid_input         ,
               r2                 ,
               N_INPUT_GLOBALPID  ,
               N_OUTPUT_GLOBALPID ,
               N_RANDOM_GLOBALPID ); 
 
    // Global Muon ID
    FLOAT_T gmuid_input [N_INPUT_GLOBALMUONID];
    FLOAT_T gmuid_output [N_OUTPUT_GLOBALMUONID];
    j = 0;

    // p, eta, nTracks, charge
    for (i = 0; i < N_INPUT_RICH; ++i)
        gmuid_input[j++] = input[i]; 
    
    // dlle, dllmu, dllk, dllp
    for (i = 0; i < N_OUTPUT_RICH; ++i)
        gmuid_input[j++] = richdll[i]; 

    // mullmu, mullbg
    for (i = 0; i < N_OUTPUT_MUON; ++i)
        gmuid_input[j++] = muondll[i]; 

    #ifdef DEBUG
    printf (" === GLOBAL MUON ID === \n");
    #endif
    gan_pipe ( gmuidTx               ,
               gmuidModel            ,
               gmuidTy               ,
               gmuid_output          ,
               gmuid_input           ,
               r3                    ,
               N_INPUT_GLOBALMUONID  ,
               N_OUTPUT_GLOBALMUONID ,
               N_RANDOM_GLOBALMUONID ); 
  
    // Format output 
    j = 0;
    for (i = 0; i < N_OUTPUT_RICH; ++i)
        output[j++] = richdll[i]; 

    for (i = 0; i < N_OUTPUT_MUON; ++i)
        output[j++] = muondll[i]; 
    
    for (i = 0; i < N_OUTPUT_GLOBALPID; ++i)
        output[j++] = gpid_output[i]; 

    for (i = 0; i < N_OUTPUT_GLOBALMUONID; ++i)
        output[j++] = gmuid_output[i];
    #ifdef DEBUG
    printf (" === OUTPUT === \n");
    for (i = 0; i < N_OUTPUT; ++i)
        printf ( "in [%d] : %.2f\n", i, output[i] );
    #endif

    return output; 
}


extern "C"
FLOAT_T* full_muon_pipe (FLOAT_T* output, const FLOAT_T *input, const FLOAT_T *random)
{
    #ifdef DEBUG
    int i = 0;
    printf ("Muon pipe: INPUT\n") ;
    for (i = 0; i < N_INPUT_RICH; ++i)
        printf ( "in [%d] : %.2f\n", i, input[i]);

    printf ("Muon pipe: RANDOM\n") ;
    for (i = 0; i < N_RANDOM_RICH; ++i)
        printf ( "rnd [%d] : %.2f\n", i, random[i]);
    #endif 

    return GenericPipe ( output , input , random ,
                         Rich_muon_tX         , Rich_muon_dnn         , Rich_muon_tY_inverse         ,
                         Muon_muon_tX         , Muon_muon_dnn         , Muon_muon_tY_inverse         ,
                         GlobalPID_muon_tX    , GlobalPID_muon_dnn    , GlobalPID_muon_tY_inverse    ,
                         GlobalMuonId_muon_tX , GlobalMuonId_muon_dnn , GlobalMuonId_muon_tY_inverse ); 
}


extern "C"
FLOAT_T* full_pion_pipe (FLOAT_T* output, const FLOAT_T *input, const FLOAT_T *random)
{
    #ifdef DEBUG
    int i = 0;
    printf ("Pion pipe: INPUT\n") ;
    for (i = 0; i < N_INPUT_RICH; ++i)
        printf ( "in [%d] : %.2f\n", i, input[i]);

    printf ("Pion pipe: RANDOM\n") ;
    for (i = 0; i < N_RANDOM_RICH; ++i)
        printf ( "rnd [%d] : %.2f\n", i, random[i]);
    #endif 

    return GenericPipe ( output , input , random ,
                         Rich_pion_tX         , Rich_pion_dnn         , Rich_pion_tY_inverse         ,
                         Muon_pion_tX         , Muon_pion_dnn         , Muon_pion_tY_inverse         ,
                         GlobalPID_pion_tX    , GlobalPID_pion_dnn    , GlobalPID_pion_tY_inverse    ,
                         GlobalMuonId_pion_tX , GlobalMuonId_pion_dnn , GlobalMuonId_pion_tY_inverse ); 
}


extern "C"
FLOAT_T* full_kaon_pipe (FLOAT_T* output, const FLOAT_T *input, const FLOAT_T *random)
{
    #ifdef DEBUG
    int i = 0;
    printf ("Kaon pipe: INPUT\n") ;
    for (i = 0; i < N_INPUT_RICH; ++i)
        printf ( "in [%d] : %.2f\n", i, input[i]);

    printf ("Kaon pipe: RANDOM\n") ;
    for (i = 0; i < N_RANDOM_RICH; ++i)
        printf ( "rnd [%d] : %.2f\n", i, random[i]);
    #endif 

    return GenericPipe ( output , input , random ,
                         Rich_kaon_tX         , Rich_kaon_dnn         , Rich_kaon_tY_inverse         ,
                         Muon_kaon_tX         , Muon_kaon_dnn         , Muon_kaon_tY_inverse         ,
                         GlobalPID_kaon_tX    , GlobalPID_kaon_dnn    , GlobalPID_kaon_tY_inverse    ,
                         GlobalMuonId_kaon_tX , GlobalMuonId_kaon_dnn , GlobalMuonId_kaon_tY_inverse ); 
}


extern "C"
FLOAT_T* full_proton_pipe (FLOAT_T* output, const FLOAT_T *input, const FLOAT_T *random)
{
    #ifdef DEBUG
    int i = 0;
    printf ("Proton pipe: INPUT\n") ;
    for (i = 0; i < N_INPUT_RICH; ++i)
        printf ( "in [%d] : %.2f\n", i, input[i]);

    printf ("Proton pipe: RANDOM\n") ;
    for (i = 0; i < N_RANDOM_RICH; ++i)
        printf ( "rnd [%d] : %.2f\n", i, random[i]);
    #endif 

    return GenericPipe ( output , input , random ,
                         Rich_proton_tX         , Rich_proton_dnn         , Rich_proton_tY_inverse         ,
                         Muon_proton_tX         , Muon_proton_dnn         , Muon_proton_tY_inverse         ,
                         GlobalPID_proton_tX    , GlobalPID_proton_dnn    , GlobalPID_proton_tY_inverse    ,
                         GlobalMuonId_proton_tX , GlobalMuonId_proton_dnn , GlobalMuonId_proton_tY_inverse ); 
}
