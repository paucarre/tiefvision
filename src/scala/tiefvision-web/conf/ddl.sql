-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

create table BOUNDING_BOX(
  file_name varchar(255) primary key,
  top int not null,
  left int not null,
  bottom int not null,
  right int not null,
  width int not null,
  height int not null
);
alter table BOUNDING_BOX add column dataset varchar(255);
update BOUNDING_BOX set dataset='UNKNOWN';

create table SIMILARITY(
  reference varchar(255),
  positive  varchar(255),
  negative  varchar(255),
  PRIMARY KEY(reference, positive, negative)
);
alter table SIMILARITY add column dataset varchar(255);
update SIMILARITY set dataset='UNKNOWN';
