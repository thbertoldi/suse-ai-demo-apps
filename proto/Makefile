.PHONY: all go python clean

all: go python

go:
	protoc --go_out=../gateway/pb --go-grpc_out=../gateway/pb \
		--go_opt=paths=source_relative --go-grpc_opt=paths=source_relative \
		demo.proto

python:
	python3 -m grpc_tools.protoc -I. \
		--python_out=../rag-service/app/generated --grpc_python_out=../rag-service/app/generated \
		demo.proto
	python3 -m grpc_tools.protoc -I. \
		--python_out=../llm-service/app/generated --grpc_python_out=../llm-service/app/generated \
		demo.proto
	python3 -m grpc_tools.protoc -I. \
		--python_out=../traffic-gen/generated --grpc_python_out=../traffic-gen/generated \
		demo.proto
	python3 -m grpc_tools.protoc -I. \
		--python_out=../agent-service/app/generated --grpc_python_out=../agent-service/app/generated \
		demo.proto
	sed -i 's/import demo_pb2/from . import demo_pb2/' ../rag-service/app/generated/demo_pb2_grpc.py
	sed -i 's/import demo_pb2/from . import demo_pb2/' ../llm-service/app/generated/demo_pb2_grpc.py
	sed -i 's/import demo_pb2/from . import demo_pb2/' ../traffic-gen/generated/demo_pb2_grpc.py
	sed -i 's/import demo_pb2/from . import demo_pb2/' ../agent-service/app/generated/demo_pb2_grpc.py

clean:
	rm -f ../gateway/pb/*.go
	rm -f ../rag-service/app/generated/demo_pb2*.py
	rm -f ../llm-service/app/generated/demo_pb2*.py
	rm -f ../traffic-gen/generated/demo_pb2*.py
	rm -f ../agent-service/app/generated/demo_pb2*.py
