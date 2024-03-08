# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: endurance.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x65ndurance.proto\x12\tendurance\"@\n\x12SharedMemoryHandle\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04size\x18\x02 \x01(\t\x12\x0e\n\x06offset\x18\x03 \x01(\t\"d\n\x06\x42itmap\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12;\n\x14shared_memory_handle\x18\x03 \x01(\x0b\x32\x1d.endurance.SharedMemoryHandle\"=\n\x0b\x42oundingBox\x12\n\n\x02x1\x18\x01 \x01(\x05\x12\n\n\x02y1\x18\x02 \x01(\x05\x12\n\n\x02x2\x18\x03 \x01(\x05\x12\n\n\x02y2\x18\x04 \x01(\x05\"P\n\x11\x41nomalyMaskParams\x12;\n\x14shared_memory_handle\x18\x01 \x01(\x0b\x32\x1d.endurance.SharedMemoryHandle\"\x8a\x01\n\x11ImageSharedMemory\x12\x17\n\x0fmodel_component\x18\x01 \x01(\t\x12!\n\x06\x62itmap\x18\x02 \x01(\x0b\x32\x11.endurance.Bitmap\x12\x39\n\x13\x61nomaly_mask_params\x18\x03 \x01(\x0b\x32\x1c.endurance.AnomalyMaskParams\"b\n\nPrediction\x12%\n\x05\x62oxes\x18\x01 \x03(\x0b\x32\x16.endurance.BoundingBox\x12\x0f\n\x07\x63lasses\x18\x02 \x03(\t\x12\x0e\n\x06scores\x18\x03 \x03(\x02\x12\x0c\n\x04name\x18\x04 \x01(\t\"\x1d\n\x0cPytorchModel\x12\r\n\x05model\x18\x01 \x01(\x0c\"$\n\x0e\x44\x65\x66\x61ultRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\"\x1f\n\rToggleRequest\x12\x0e\n\x06\x61\x63tive\x18\x01 \x01(\x08\"\xd0\x01\n\x0ePredictRequest\x12+\n\x05image\x18\x01 \x01(\x0b\x32\x1c.endurance.ImageSharedMemory\x12\x12\n\nmodel_name\x18\x02 \x01(\t\x12\x17\n\x0finvert_channels\x18\x03 \x01(\x08\x12\x35\n\x06scores\x18\x04 \x03(\x0b\x32%.endurance.PredictRequest.ScoresEntry\x1a-\n\x0bScoresEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"a\n\x0f\x41\x64\x64ModelRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x12\n\nmodel_info\x18\x02 \x01(\x0c\x12&\n\x05model\x18\x03 \x01(\x0b\x32\x17.endurance.PytorchModel\"\x8e\x01\n\x10LoadModelRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x37\n\x06scores\x18\x02 \x03(\x0b\x32\'.endurance.LoadModelRequest.ScoresEntry\x1a-\n\x0bScoresEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"\x13\n\x11ListModelsRequest\"\x1f\n\x0f\x44\x65\x66\x61ultResponse\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\"<\n\x0fPredictResponse\x12)\n\nprediction\x18\x01 \x01(\x0b\x32\x15.endurance.Prediction\"$\n\x12ListModelsResponse\x12\x0e\n\x06models\x18\x01 \x01(\t2\xba\x03\n\x10\x45nduranceService\x12@\n\x07predict\x12\x19.endurance.PredictRequest\x1a\x1a.endurance.PredictResponse\x12\x43\n\tadd_model\x12\x1a.endurance.AddModelRequest\x1a\x1a.endurance.DefaultResponse\x12\x45\n\nload_model\x12\x1b.endurance.LoadModelRequest\x1a\x1a.endurance.DefaultResponse\x12\x45\n\x0c\x64\x65lete_model\x12\x19.endurance.DefaultRequest\x1a\x1a.endurance.DefaultResponse\x12J\n\x0blist_models\x12\x1c.endurance.ListModelsRequest\x1a\x1d.endurance.ListModelsResponse\x12\x45\n\rtoggle_logger\x12\x18.endurance.ToggleRequest\x1a\x1a.endurance.DefaultResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'endurance_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_PREDICTREQUEST_SCORESENTRY']._options = None
  _globals['_PREDICTREQUEST_SCORESENTRY']._serialized_options = b'8\001'
  _globals['_LOADMODELREQUEST_SCORESENTRY']._options = None
  _globals['_LOADMODELREQUEST_SCORESENTRY']._serialized_options = b'8\001'
  _globals['_SHAREDMEMORYHANDLE']._serialized_start=30
  _globals['_SHAREDMEMORYHANDLE']._serialized_end=94
  _globals['_BITMAP']._serialized_start=96
  _globals['_BITMAP']._serialized_end=196
  _globals['_BOUNDINGBOX']._serialized_start=198
  _globals['_BOUNDINGBOX']._serialized_end=259
  _globals['_ANOMALYMASKPARAMS']._serialized_start=261
  _globals['_ANOMALYMASKPARAMS']._serialized_end=341
  _globals['_IMAGESHAREDMEMORY']._serialized_start=344
  _globals['_IMAGESHAREDMEMORY']._serialized_end=482
  _globals['_PREDICTION']._serialized_start=484
  _globals['_PREDICTION']._serialized_end=582
  _globals['_PYTORCHMODEL']._serialized_start=584
  _globals['_PYTORCHMODEL']._serialized_end=613
  _globals['_DEFAULTREQUEST']._serialized_start=615
  _globals['_DEFAULTREQUEST']._serialized_end=651
  _globals['_TOGGLEREQUEST']._serialized_start=653
  _globals['_TOGGLEREQUEST']._serialized_end=684
  _globals['_PREDICTREQUEST']._serialized_start=687
  _globals['_PREDICTREQUEST']._serialized_end=895
  _globals['_PREDICTREQUEST_SCORESENTRY']._serialized_start=850
  _globals['_PREDICTREQUEST_SCORESENTRY']._serialized_end=895
  _globals['_ADDMODELREQUEST']._serialized_start=897
  _globals['_ADDMODELREQUEST']._serialized_end=994
  _globals['_LOADMODELREQUEST']._serialized_start=997
  _globals['_LOADMODELREQUEST']._serialized_end=1139
  _globals['_LOADMODELREQUEST_SCORESENTRY']._serialized_start=850
  _globals['_LOADMODELREQUEST_SCORESENTRY']._serialized_end=895
  _globals['_LISTMODELSREQUEST']._serialized_start=1141
  _globals['_LISTMODELSREQUEST']._serialized_end=1160
  _globals['_DEFAULTRESPONSE']._serialized_start=1162
  _globals['_DEFAULTRESPONSE']._serialized_end=1193
  _globals['_PREDICTRESPONSE']._serialized_start=1195
  _globals['_PREDICTRESPONSE']._serialized_end=1255
  _globals['_LISTMODELSRESPONSE']._serialized_start=1257
  _globals['_LISTMODELSRESPONSE']._serialized_end=1293
  _globals['_ENDURANCESERVICE']._serialized_start=1296
  _globals['_ENDURANCESERVICE']._serialized_end=1738
# @@protoc_insertion_point(module_scope)
