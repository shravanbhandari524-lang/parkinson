'use strict';

/** Shared test fixtures. ML responses here mirror the real ML service. */

function mlPredictionResponse(modalities) {
  const body = {
    multimodal: {
      probability: 0.72,
      top_features: modalities.map((m) => ({
        modality: m,
        feature: `${m}_feature_1`,
        contribution: 0.31,
      })),
    },
    risk_band: 'high',
    model_version: '1.0.0',
  };
  for (const modality of modalities) {
    body[modality] = {
      probability: 0.68,
      top_features: [{ modality, feature: `${modality}_feature_1`, contribution: 0.31 }],
    };
  }
  return body;
}

function bufferFor(modality) {
  const contents = {
    voice: 'MDVP:Fo(Hz),MDVP:Fhi(Hz)\n152.125,197.2',
    handwriting: 'RMS,MAX_BETWEEN_STHT\n1.2,0.4',
    gait: '1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14\t15\t16\t17\t18\t19\n',
  };
  return Buffer.from(contents[modality] || 'x');
}

module.exports = { mlPredictionResponse, bufferFor };
