-- Create a table for langchain Supabase documents and migrate data of original table records
       create table documents_langchain (
       id bigserial primary key,
       content text, -- corresponds to Document.pageContent
       metadata jsonb, -- corresponds to Document.metadata
       embedding vector(1536) -- 1536 works for OpenAI embeddings, change if needed
       );

       CREATE FUNCTION match_documents_langchain(query_embedding vector(1536), match_count int)
           RETURNS TABLE(
               id bigint,
               content text,
               metadata jsonb,
               -- we return matched vectors to enable maximal marginal relevance searches
               embedding vector(1536),
               similarity float)
           LANGUAGE plpgsql
           AS $$
           # variable_conflict use_column
       BEGIN
           RETURN query
           SELECT
               id,
               content,
               metadata,
               embedding,
               1 -(documents_langchain.embedding <=> query_embedding) AS similarity
           FROM
               documents_langchain
           ORDER BY
               documents_langchain.embedding <=> query_embedding
           LIMIT match_count;
       END;
       $$;

-- Copy & migrate data of original table records
       insert into documents_langchain ( content, embedding, metadata) select content, embedding,  json_build_object('source', url) from documents 