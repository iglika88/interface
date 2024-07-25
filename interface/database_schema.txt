CREATE TABLE alembic_version (
	version_num VARCHAR(32) NOT NULL, 
	CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
CREATE TABLE users (
	id INTEGER NOT NULL, 
	username VARCHAR(20) NOT NULL, 
	email VARCHAR(120) NOT NULL, 
	password_hash VARCHAR(128) NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (email), 
	UNIQUE (username)
);
CREATE TABLE vocabulary_entries (
	id INTEGER NOT NULL, 
	item VARCHAR(255) NOT NULL, 
	pos VARCHAR(50), 
	translation VARCHAR(255), 
	lesson_title VARCHAR(255), 
	reading_or_listening VARCHAR(50), 
	course_code VARCHAR(50) NOT NULL, 
	cefr_level VARCHAR(10) NOT NULL, 
	domain VARCHAR(255) NOT NULL, 
	user VARCHAR(50) NOT NULL, 
	date_loaded DATETIME NOT NULL, 
	number_of_contexts INTEGER NOT NULL, 
	PRIMARY KEY (id)
);
CREATE TABLE context_entry (
	id INTEGER NOT NULL, 
	item_id INTEGER NOT NULL, 
	context TEXT NOT NULL, 
	user VARCHAR(50) NOT NULL, 
	date_added DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(item_id) REFERENCES vocabulary_entries (id)
);
