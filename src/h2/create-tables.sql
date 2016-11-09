-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

create table if not exists BOUNDING_BOX(
  file_name varchar(255) primary key,
  top int not null,
  left int not null,
  bottom int not null,
  right int not null,
  width int not null,
  height int not null,
  dataset varchar(255) default 'UNKNOWN'
);

create table if not exists SIMILARITY(
  reference varchar(255),
  positive  varchar(255),
  negative  varchar(255),
  dataset varchar(255) default 'UNKNOWN',
  PRIMARY KEY(reference, positive, negative)
);
